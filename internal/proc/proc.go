// Package proc starts engine servers in their own process group and stops the whole
// group, so child processes (e.g. Ollama runners) never outlive a run.
package proc

import (
	"context"
	"fmt"
	"io"
	"net"
	"os"
	"os/exec"
	"strings"
	"time"
)

// Spec describes a process to start.
type Spec struct {
	Path    string
	Args    []string
	Env     map[string]string // added to the inherited environment
	Dir     string
	LogPath string // stdout+stderr are appended here
}

// Proc is a running process group.
type Proc struct {
	Cmd     *exec.Cmd
	PGID    int
	LogPath string
	Started time.Time

	done    chan struct{}
	waitErr error
	log     *os.File
}

// Start launches spec in a new process group.
func Start(spec Spec) (*Proc, error) {
	cmd := exec.Command(spec.Path, spec.Args...)
	cmd.Dir = spec.Dir
	cmd.Env = os.Environ()
	for k, v := range spec.Env {
		cmd.Env = append(cmd.Env, k+"="+v)
	}
	cmd.SysProcAttr = newGroupAttr()
	var logf *os.File
	if spec.LogPath != "" {
		f, err := os.OpenFile(spec.LogPath, os.O_CREATE|os.O_WRONLY|os.O_APPEND, 0o644)
		if err != nil {
			return nil, err
		}
		logf = f
		cmd.Stdout, cmd.Stderr = f, f
	} else {
		cmd.Stdout, cmd.Stderr = io.Discard, io.Discard
	}
	if err := cmd.Start(); err != nil {
		if logf != nil {
			logf.Close()
		}
		return nil, fmt.Errorf("start %s: %w", spec.Path, err)
	}
	p := &Proc{Cmd: cmd, PGID: cmd.Process.Pid, LogPath: spec.LogPath, Started: time.Now(), done: make(chan struct{}), log: logf}
	go func() {
		p.waitErr = cmd.Wait()
		if p.log != nil {
			p.log.Close()
		}
		close(p.done)
	}()
	return p, nil
}

// Done is closed when the leader process exits.
func (p *Proc) Done() <-chan struct{} { return p.done }

// Exited reports whether the leader has exited, and its error.
func (p *Proc) Exited() (bool, error) {
	select {
	case <-p.done:
		return true, p.waitErr
	default:
		return false, nil
	}
}

// Stop sends SIGTERM to the group, waits up to grace, then SIGKILLs the group. It
// returns once the leader has exited and no process in the group remains.
func (p *Proc) Stop(grace time.Duration) error {
	terminateGroup(p.PGID)
	select {
	case <-p.done:
	case <-time.After(grace):
		KillGroup(p.PGID)
		<-p.done
	}
	// Children may linger after the leader exits; make sure the group is gone.
	deadline := time.Now().Add(grace)
	for groupAlive(p.PGID) {
		if time.Now().After(deadline) {
			KillGroup(p.PGID)
			deadline = time.Now().Add(5 * time.Second)
		}
		time.Sleep(50 * time.Millisecond)
	}
	return nil
}

// LogTail returns the last n lines of the process log, for error messages.
func (p *Proc) LogTail(n int) string {
	if p.LogPath == "" {
		return ""
	}
	b, err := os.ReadFile(p.LogPath)
	if err != nil {
		return ""
	}
	lines := strings.Split(strings.TrimRight(string(b), "\n"), "\n")
	if len(lines) > n {
		lines = lines[len(lines)-n:]
	}
	return strings.Join(lines, "\n")
}

// FreePort returns a currently free TCP port on 127.0.0.1.
func FreePort() (int, error) {
	l, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		return 0, err
	}
	defer l.Close()
	return l.Addr().(*net.TCPAddr).Port, nil
}

// Run executes a short-lived command to completion, appending output to logPath.
func Run(ctx context.Context, spec Spec) error {
	cmd := exec.CommandContext(ctx, spec.Path, spec.Args...)
	cmd.Dir = spec.Dir
	cmd.Env = os.Environ()
	for k, v := range spec.Env {
		cmd.Env = append(cmd.Env, k+"="+v)
	}
	var out io.Writer = io.Discard
	if spec.LogPath != "" {
		f, err := os.OpenFile(spec.LogPath, os.O_CREATE|os.O_WRONLY|os.O_APPEND, 0o644)
		if err != nil {
			return err
		}
		defer f.Close()
		fmt.Fprintf(f, "\n$ %s %s\n", spec.Path, strings.Join(spec.Args, " "))
		out = f
	}
	cmd.Stdout, cmd.Stderr = out, out
	if err := cmd.Run(); err != nil {
		return fmt.Errorf("%s %s: %w", spec.Path, strings.Join(spec.Args, " "), err)
	}
	return nil
}
