package install

import (
	"github.com/thisisadityapatel/kepler/internal/download"
	"github.com/thisisadityapatel/kepler/internal/events"
)

func progressPublisher(env Env) download.Progress {
	return func(done, total int64) {
		env.Bus.Publish(events.Event{Type: events.DownloadProgress, Engine: env.Engine, Done: done, Total: total})
	}
}
