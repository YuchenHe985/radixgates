// Package breaker implements a per-backend circuit breaker.
//
// States: Closed (traffic flows) -> Open (traffic blocked for a cool-down after
// FailureThreshold consecutive failures) -> HalfOpen (a bounded number of probe
// requests are let through) -> Closed on probe success, or Open again with a
// doubled cool-down (capped) on probe failure.
//
// Admission returns a Ticket that remembers the breaker "generation" it was
// issued under. A result that arrives after the breaker has changed state (for
// example a slow request that started while Closed and finishes while Open) is
// stale and ignored, so it cannot close a breaker it never probed.
package breaker

import (
	"sync"
	"time"
)

type State int

const (
	Closed State = iota
	Open
	HalfOpen
)

func (s State) String() string {
	switch s {
	case Closed:
		return "closed"
	case Open:
		return "open"
	case HalfOpen:
		return "half_open"
	}
	return "unknown"
}

type Config struct {
	FailureThreshold int
	OpenDuration     time.Duration
	MaxOpenDuration  time.Duration
	HalfOpenMax      int
}

// Ticket is returned by Allow and must be handed back to Done or Cancel.
type Ticket struct {
	probe bool
	gen   uint64
}

type Option func(*Breaker)

// WithClock injects a clock (tests).
func WithClock(now func() time.Time) Option { return func(b *Breaker) { b.now = now } }

// WithOnChange registers a state-change callback. It runs while the breaker
// lock is held, so it must be fast and must not call back into the breaker.
func WithOnChange(f func(from, to State)) Option { return func(b *Breaker) { b.onChange = f } }

type Breaker struct {
	mu           sync.Mutex
	cfg          Config
	now          func() time.Time
	onChange     func(from, to State)
	state        State
	gen          uint64
	consecFail   int
	openUntil    time.Time
	curOpen      time.Duration
	halfInFlight int
}

func New(cfg Config, opts ...Option) *Breaker {
	if cfg.FailureThreshold < 1 {
		cfg.FailureThreshold = 1
	}
	if cfg.HalfOpenMax < 1 {
		cfg.HalfOpenMax = 1
	}
	if cfg.OpenDuration <= 0 {
		cfg.OpenDuration = time.Second
	}
	if cfg.MaxOpenDuration < cfg.OpenDuration {
		cfg.MaxOpenDuration = cfg.OpenDuration
	}
	b := &Breaker{cfg: cfg, now: time.Now, curOpen: cfg.OpenDuration}
	for _, o := range opts {
		o(b)
	}
	return b
}

func (b *Breaker) setLocked(to State) {
	if b.state == to {
		return
	}
	from := b.state
	b.state = to
	b.gen++
	if b.onChange != nil {
		b.onChange(from, to)
	}
}

// advanceLocked lazily promotes Open to HalfOpen once the cool-down has passed.
func (b *Breaker) advanceLocked() {
	if b.state == Open && !b.now().Before(b.openUntil) {
		b.halfInFlight = 0
		b.setLocked(HalfOpen)
	}
}

func (b *Breaker) tripLocked() {
	b.consecFail = 0
	b.openUntil = b.now().Add(b.curOpen)
	b.setLocked(Open)
}

// State returns the current state.
func (b *Breaker) State() State {
	b.mu.Lock()
	defer b.mu.Unlock()
	b.advanceLocked()
	return b.state
}

// Available reports whether Allow would currently succeed, without consuming a probe slot.
func (b *Breaker) Available() bool {
	b.mu.Lock()
	defer b.mu.Unlock()
	b.advanceLocked()
	switch b.state {
	case Closed:
		return true
	case HalfOpen:
		return b.halfInFlight < b.cfg.HalfOpenMax
	}
	return false
}

// Allow admits one request. In HalfOpen it consumes a probe slot.
func (b *Breaker) Allow() (Ticket, bool) {
	b.mu.Lock()
	defer b.mu.Unlock()
	b.advanceLocked()
	switch b.state {
	case Closed:
		return Ticket{gen: b.gen}, true
	case HalfOpen:
		if b.halfInFlight < b.cfg.HalfOpenMax {
			b.halfInFlight++
			return Ticket{probe: true, gen: b.gen}, true
		}
	}
	return Ticket{}, false
}

// Done records the outcome of an admitted request.
func (b *Breaker) Done(t Ticket, success bool) {
	b.mu.Lock()
	defer b.mu.Unlock()
	b.advanceLocked()
	if t.gen != b.gen {
		return // stale: issued under an earlier state
	}
	switch b.state {
	case Closed:
		if success {
			b.consecFail = 0
			return
		}
		b.consecFail++
		if b.consecFail >= b.cfg.FailureThreshold {
			b.tripLocked()
		}
	case HalfOpen:
		if !t.probe {
			return
		}
		if success {
			b.consecFail = 0
			b.curOpen = b.cfg.OpenDuration
			b.setLocked(Closed)
			return
		}
		b.curOpen = min(b.curOpen*2, b.cfg.MaxOpenDuration)
		b.tripLocked()
	}
}

// Cancel returns a probe slot without a verdict (client went away, backend said "busy").
func (b *Breaker) Cancel(t Ticket) {
	b.mu.Lock()
	defer b.mu.Unlock()
	if t.probe && b.state == HalfOpen && t.gen == b.gen && b.halfInFlight > 0 {
		b.halfInFlight--
	}
}
