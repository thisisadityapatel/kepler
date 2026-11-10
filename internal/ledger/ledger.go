// Package ledger records every artifact Kepler creates so it can always be cleaned up,
// including after a crash. Rows are written before the artifact exists (write-ahead);
// the reaper deletes temp artifacts whose owning process is dead.
package ledger

import (
	"context"
	"database/sql"
	"fmt"
	"os"
	"time"

	_ "modernc.org/sqlite"

	"github.com/thisisadityapatel/kepler/internal/home"
	"github.com/thisisadityapatel/kepler/internal/proc"
)

// Artifact kinds.
const (
	KindFile          = "file"
	KindDir           = "dir"
	KindProcessGroup  = "process_group"
	KindEngineInstall = "engine_install"
)

// States.
const (
	StateTemp     = "temp"     // delete when owner is done or dead
	StateKept     = "kept"     // user asked to keep (--keep); listed by `kepler clean`
	StateReleased = "released" // ownership handed to the registry (installed engine, canary)
	StateDeleted  = "deleted"
)

// Artifact is one ledger row.
type Artifact struct {
	ID        int64
	RunID     string
	Kind      string
	Path      string
	PGID      int
	OwnerPID  int
	State     string
	CreatedAt time.Time
	Note      string
}

// Ledger is the SQLite-backed artifact ledger.
type Ledger struct {
	db    *sql.DB
	paths home.Paths
}

const schema = `
CREATE TABLE IF NOT EXISTS artifacts (
  id          INTEGER PRIMARY KEY,
  run_id      TEXT NOT NULL,
  kind        TEXT NOT NULL,
  path        TEXT NOT NULL DEFAULT '',
  pgid        INTEGER NOT NULL DEFAULT 0,
  owner_pid   INTEGER NOT NULL,
  state       TEXT NOT NULL,
  created_at  TEXT NOT NULL,
  note        TEXT NOT NULL DEFAULT ''
);
CREATE INDEX IF NOT EXISTS artifacts_state ON artifacts(state);`

// Open opens (creating if needed) the ledger at paths.Ledger.
func Open(paths home.Paths) (*Ledger, error) {
	if err := os.MkdirAll(paths.State, 0o755); err != nil {
		return nil, err
	}
	db, err := sql.Open("sqlite", "file:"+paths.Ledger+"?_pragma=busy_timeout(5000)&_pragma=journal_mode(WAL)")
	if err != nil {
		return nil, err
	}
	db.SetMaxOpenConns(1)
	if _, err := db.Exec(schema); err != nil {
		db.Close()
		return nil, fmt.Errorf("init ledger: %w", err)
	}
	return &Ledger{db: db, paths: paths}, nil
}

// Close closes the database.
func (l *Ledger) Close() error { return l.db.Close() }

// Record inserts a temp row owned by this process. Call it BEFORE creating the artifact.
// File and dir paths must be inside KEPLER_HOME.
func (l *Ledger) Record(ctx context.Context, a Artifact) (int64, error) {
	if (a.Kind == KindFile || a.Kind == KindDir || a.Kind == KindEngineInstall) && !l.paths.Contains(a.Path) {
		return 0, fmt.Errorf("%w: %s", home.ErrOutsideHome, a.Path)
	}
	if a.OwnerPID == 0 {
		a.OwnerPID = os.Getpid()
	}
	res, err := l.db.ExecContext(ctx,
		`INSERT INTO artifacts(run_id, kind, path, pgid, owner_pid, state, created_at, note) VALUES (?,?,?,?,?,?,?,?)`,
		a.RunID, a.Kind, a.Path, a.PGID, a.OwnerPID, StateTemp, time.Now().UTC().Format(time.RFC3339Nano), a.Note)
	if err != nil {
		return 0, err
	}
	return res.LastInsertId()
}

// SetPGID records the process group once the process has started.
func (l *Ledger) SetPGID(ctx context.Context, id int64, pgid int) error {
	_, err := l.db.ExecContext(ctx, `UPDATE artifacts SET pgid=? WHERE id=?`, pgid, id)
	return err
}

func (l *Ledger) setState(ctx context.Context, id int64, state string) error {
	_, err := l.db.ExecContext(ctx, `UPDATE artifacts SET state=? WHERE id=?`, state, id)
	return err
}

// Release hands ownership to the registry; the reaper will never touch the row again.
func (l *Ledger) Release(ctx context.Context, id int64) error {
	return l.setState(ctx, id, StateReleased)
}

// Keep marks a temp artifact as user-kept.
func (l *Ledger) Keep(ctx context.Context, id int64) error { return l.setState(ctx, id, StateKept) }

// Dispose deletes the artifact now (path or process group) and marks the row deleted.
func (l *Ledger) Dispose(ctx context.Context, id int64) error {
	a, err := l.get(ctx, id)
	if err != nil {
		return err
	}
	if err := l.destroy(a); err != nil {
		return err
	}
	return l.setState(ctx, id, StateDeleted)
}

// List returns rows in the given state (all states if state is "").
func (l *Ledger) List(ctx context.Context, state string) ([]Artifact, error) {
	q := `SELECT id, run_id, kind, path, pgid, owner_pid, state, created_at, note FROM artifacts`
	var args []any
	if state != "" {
		q += ` WHERE state=?`
		args = append(args, state)
	}
	rows, err := l.db.QueryContext(ctx, q+` ORDER BY id`, args...)
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	var out []Artifact
	for rows.Next() {
		var a Artifact
		var created string
		if err := rows.Scan(&a.ID, &a.RunID, &a.Kind, &a.Path, &a.PGID, &a.OwnerPID, &a.State, &created, &a.Note); err != nil {
			return nil, err
		}
		a.CreatedAt, _ = time.Parse(time.RFC3339Nano, created)
		out = append(out, a)
	}
	return out, rows.Err()
}

func (l *Ledger) get(ctx context.Context, id int64) (Artifact, error) {
	var a Artifact
	var created string
	err := l.db.QueryRowContext(ctx,
		`SELECT id, run_id, kind, path, pgid, owner_pid, state, created_at, note FROM artifacts WHERE id=?`, id).
		Scan(&a.ID, &a.RunID, &a.Kind, &a.Path, &a.PGID, &a.OwnerPID, &a.State, &created, &a.Note)
	a.CreatedAt, _ = time.Parse(time.RFC3339Nano, created)
	return a, err
}

// ReapReport summarizes a reaper pass.
type ReapReport struct {
	Reaped []Artifact
	Errors []error
}

// Reap disposes every temp artifact whose owner process is no longer alive. If all is
// true, temp artifacts of live owners other than this process are also left alone, but
// kept artifacts are disposed too (`kepler clean --all`).
func (l *Ledger) Reap(ctx context.Context, includeKept bool) (ReapReport, error) {
	var rep ReapReport
	rows, err := l.List(ctx, "")
	if err != nil {
		return rep, err
	}
	self := os.Getpid()
	for _, a := range rows {
		switch {
		case a.State == StateTemp && a.OwnerPID != self && !proc.Alive(a.OwnerPID):
		case a.State == StateKept && includeKept:
		default:
			continue
		}
		if err := l.destroy(a); err != nil {
			rep.Errors = append(rep.Errors, fmt.Errorf("artifact %d (%s %s): %w", a.ID, a.Kind, a.Path, err))
			continue
		}
		if err := l.setState(ctx, a.ID, StateDeleted); err != nil {
			rep.Errors = append(rep.Errors, err)
			continue
		}
		rep.Reaped = append(rep.Reaped, a)
	}
	return rep, nil
}

func (l *Ledger) destroy(a Artifact) error {
	switch a.Kind {
	case KindProcessGroup:
		if a.PGID > 0 {
			proc.KillGroup(a.PGID)
		}
		return nil
	default:
		if a.Path == "" {
			return nil
		}
		return l.paths.SafeRemoveAll(a.Path)
	}
}
