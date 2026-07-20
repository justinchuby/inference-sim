import { describe, expect, it } from "vitest";
import {
  DiscreteEventSimulator,
  SimulationError,
} from "../src/index.js";

describe("DiscreteEventSimulator", () => {
  it("orders events by timestamp and stable source sequence", () => {
    const simulator = new DiscreteEventSimulator<string>();
    const observed: string[] = [];

    simulator.scheduleAt(10, "a");
    simulator.scheduleAt(5, "b");
    simulator.scheduleAt(10, "c");

    const result = simulator.run((event, simulation) => {
      observed.push(event.payload);
      if (event.payload === "a") {
        simulation.scheduleAt(simulation.nowNs, "d");
      }
    });

    expect(observed).toEqual(["b", "a", "c", "d"]);
    expect(result).toEqual({
      processedEvents: 4,
      skippedCancelledEvents: 0,
      nowNs: 10,
      pendingEvents: 0,
    });
  });

  it("supports cancellation and bounded partial runs", () => {
    const simulator = new DiscreteEventSimulator<string>();
    const cancelled = simulator.scheduleAt(2, "cancelled");
    simulator.scheduleAt(4, "later");

    expect(simulator.cancel(cancelled)).toBe(true);
    expect(simulator.cancel(cancelled)).toBe(false);

    const first = simulator.run(() => {
      throw new Error("cancelled event must not run");
    }, { untilNs: 3 });
    expect(first.skippedCancelledEvents).toBe(1);
    expect(first.pendingEvents).toBe(1);
    expect(first.nowNs).toBe(0);

    const observed: string[] = [];
    const second = simulator.run((event) => observed.push(event.payload));
    expect(observed).toEqual(["later"]);
    expect(second.nowNs).toBe(4);
  });

  it("advances exactly one non-cancelled event for streaming runtimes", () => {
    const simulator = new DiscreteEventSimulator<string>();
    const cancelled = simulator.scheduleAt(1, "cancelled");
    simulator.scheduleAt(2, "first");
    simulator.scheduleAt(3, "second");
    simulator.cancel(cancelled);
    const observed: string[] = [];

    expect(simulator.runNext((event) => observed.push(event.payload))).toBe(true);
    expect(observed).toEqual(["first"]);
    expect(simulator.nowNs).toBe(2);
    expect(simulator.pendingCount).toBe(1);
    expect(simulator.runNext((event) => observed.push(event.payload))).toBe(true);
    expect(simulator.runNext((event) => observed.push(event.payload))).toBe(false);
    expect(observed).toEqual(["first", "second"]);
  });

  it("rejects scheduling into the past", () => {
    const simulator = new DiscreteEventSimulator<string>();
    simulator.scheduleAt(5, "now");

    expect(() => simulator.run((_event, simulation) => {
      simulation.scheduleAt(4, "past");
    })).toThrowError(SimulationError);
  });

  it("stops self-scheduling traces at the event budget", () => {
    const simulator = new DiscreteEventSimulator<string>();
    simulator.scheduleAt(0, "repeat");

    expect(() => simulator.run((_event, simulation) => {
      simulation.scheduleAfter(0, "repeat");
    }, { maxEvents: 3 })).toThrowError(
      "simulation exceeded maximum event count 3",
    );
    expect(simulator.pendingCount).toBe(1);
  });
});
