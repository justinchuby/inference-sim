export interface ScheduledEvent<E> {
  readonly id: number;
  readonly timestampNs: number;
  readonly sequence: number;
  readonly payload: E;
}

export interface SimulationRunOptions {
  readonly untilNs?: number;
  readonly maxEvents?: number;
}

export interface SimulationRunResult {
  readonly processedEvents: number;
  readonly skippedCancelledEvents: number;
  readonly nowNs: number;
  readonly pendingEvents: number;
}

export type SimulationEventHandler<E> = (
  event: ScheduledEvent<E>,
  simulator: DiscreteEventSimulator<E>,
) => void;

export class SimulationError extends Error {
  constructor(message: string) {
    super(message);
    this.name = "SimulationError";
  }
}

/**
 * Deterministic single-threaded discrete-event kernel.
 *
 * Events are ordered by `(timestampNs, sequence)`. Handlers may schedule more
 * events at the current time, but never in the past.
 */
export class DiscreteEventSimulator<E> {
  private readonly heap: ScheduledEvent<E>[] = [];
  private readonly pendingIds = new Set<number>();
  private readonly cancelledIds = new Set<number>();
  private nextId = 1;
  private nextSequence = 0;
  private currentTimeNs = 0;

  get nowNs(): number {
    return this.currentTimeNs;
  }

  get pendingCount(): number {
    return this.pendingIds.size;
  }

  scheduleAt(timestampNs: number, payload: E): number {
    assertSafeNonNegativeInteger(timestampNs, "event timestamp");
    if (timestampNs < this.currentTimeNs) {
      throw new SimulationError(
        `cannot schedule event at ${timestampNs}ns before current time ${this.currentTimeNs}ns`,
      );
    }
    if (!Number.isSafeInteger(this.nextId) || !Number.isSafeInteger(this.nextSequence)) {
      throw new SimulationError("event identity space exhausted");
    }

    const event: ScheduledEvent<E> = {
      id: this.nextId++,
      timestampNs,
      sequence: this.nextSequence++,
      payload,
    };
    this.pendingIds.add(event.id);
    this.push(event);
    return event.id;
  }

  scheduleAfter(delayNs: number, payload: E): number {
    assertSafeNonNegativeInteger(delayNs, "event delay");
    const timestampNs = this.currentTimeNs + delayNs;
    if (!Number.isSafeInteger(timestampNs)) {
      throw new SimulationError("event timestamp exceeds Number.MAX_SAFE_INTEGER");
    }
    return this.scheduleAt(timestampNs, payload);
  }

  cancel(eventId: number): boolean {
    assertSafePositiveInteger(eventId, "event id");
    if (!this.pendingIds.has(eventId) || this.cancelledIds.has(eventId)) {
      return false;
    }
    this.cancelledIds.add(eventId);
    return true;
  }

  run(
    handler: SimulationEventHandler<E>,
    options: SimulationRunOptions = {},
  ): SimulationRunResult {
    const untilNs = options.untilNs ?? Number.MAX_SAFE_INTEGER;
    const maxEvents = options.maxEvents ?? 1_000_000;
    assertSafeNonNegativeInteger(untilNs, "run deadline");
    assertSafePositiveInteger(maxEvents, "maximum event count");

    let processedEvents = 0;
    let skippedCancelledEvents = 0;

    while (this.heap.length > 0) {
      const next = this.heap[0];
      if (next.timestampNs > untilNs) {
        break;
      }

      if (this.cancelledIds.has(next.id)) {
        const cancelled = this.pop();
        this.pendingIds.delete(cancelled.id);
        this.cancelledIds.delete(cancelled.id);
        skippedCancelledEvents++;
        continue;
      }
      if (processedEvents >= maxEvents) {
        throw new SimulationError(
          `simulation exceeded maximum event count ${maxEvents} at ${this.currentTimeNs}ns`,
        );
      }

      const event = this.pop();
      this.pendingIds.delete(event.id);

      this.currentTimeNs = event.timestampNs;
      processedEvents++;
      handler(event, this);
    }

    return {
      processedEvents,
      skippedCancelledEvents,
      nowNs: this.currentTimeNs,
      pendingEvents: this.pendingIds.size,
    };
  }

  private push(event: ScheduledEvent<E>): void {
    this.heap.push(event);
    let index = this.heap.length - 1;
    while (index > 0) {
      const parent = Math.floor((index - 1) / 2);
      if (compareEvents(this.heap[parent], event) <= 0) {
        break;
      }
      this.heap[index] = this.heap[parent];
      index = parent;
    }
    this.heap[index] = event;
  }

  private pop(): ScheduledEvent<E> {
    const first = this.heap[0];
    const last = this.heap.pop();
    if (this.heap.length === 0 || !last) {
      return first;
    }

    let index = 0;
    while (true) {
      const left = index * 2 + 1;
      const right = left + 1;
      if (left >= this.heap.length) {
        break;
      }
      let child = left;
      if (
        right < this.heap.length
        && compareEvents(this.heap[right], this.heap[left]) < 0
      ) {
        child = right;
      }
      if (compareEvents(last, this.heap[child]) <= 0) {
        break;
      }
      this.heap[index] = this.heap[child];
      index = child;
    }
    this.heap[index] = last;
    return first;
  }
}

function compareEvents<E>(left: ScheduledEvent<E>, right: ScheduledEvent<E>): number {
  if (left.timestampNs !== right.timestampNs) {
    return left.timestampNs - right.timestampNs;
  }
  return left.sequence - right.sequence;
}

function assertSafeNonNegativeInteger(value: number, label: string): void {
  if (!Number.isSafeInteger(value) || value < 0) {
    throw new SimulationError(`${label} must be a non-negative safe integer; got ${value}`);
  }
}

function assertSafePositiveInteger(value: number, label: string): void {
  if (!Number.isSafeInteger(value) || value <= 0) {
    throw new SimulationError(`${label} must be a positive safe integer; got ${value}`);
  }
}
