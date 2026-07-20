import {
  SCENARIO_SCHEMA_VERSION,
  type AllocationClass,
  type AllocationReservation,
  type MemoryDomainSpec,
  type ScenarioMemoryLedgerEntry,
  type ScenarioValidationIssue,
  type ScenarioValidationResult,
  type SimLinkSpec,
  type SimulationScenario,
  type TransferRequirement,
} from "./scenario-types.js";

export class ScenarioValidationError extends Error {
  readonly issues: readonly ScenarioValidationIssue[];

  constructor(issues: readonly ScenarioValidationIssue[]) {
    super(
      `scenario validation failed with ${issues.length} issue(s): ${
        issues.map((issue) => `${issue.path}: ${issue.message}`).join("; ")
      }`,
    );
    this.name = "ScenarioValidationError";
    this.issues = issues;
  }
}

export function assertValidScenario(scenario: SimulationScenario): void {
  const result = validateScenario(scenario);
  if (!result.valid) {
    throw new ScenarioValidationError(result.issues);
  }
}

export function validateScenario(
  scenario: SimulationScenario,
): ScenarioValidationResult {
  const issues: ScenarioValidationIssue[] = [];
  const add = (code: string, path: string, message: string): void => {
    issues.push({ code, path, message });
  };

  if (scenario.schemaVersion !== SCENARIO_SCHEMA_VERSION) {
    add(
      "schema_version",
      "schemaVersion",
      `expected ${SCENARIO_SCHEMA_VERSION}, got ${scenario.schemaVersion}`,
    );
  }
  if (scenario.id.length === 0) {
    add("empty_id", "id", "scenario id must not be empty");
  }

  validatePositiveInteger(
    scenario.workload.batchSize,
    "workload.batchSize",
    add,
  );
  validateNonNegativeInteger(
    scenario.workload.inputSequenceLength,
    "workload.inputSequenceLength",
    add,
  );
  validateNonNegativeInteger(
    scenario.workload.outputSequenceLength,
    "workload.outputSequenceLength",
    add,
  );
  if (scenario.workload.speculative) {
    validateNonNegativeInteger(
      scenario.workload.speculative.maxAdditionalTokens,
      "workload.speculative.maxAdditionalTokens",
      add,
    );
  }
  validateNonNegativeInteger(
    scenario.execution.topologyEpoch,
    "execution.topologyEpoch",
    add,
  );
  validateNonNegativeInteger(scenario.execution.seed, "execution.seed", add);
  validatePositiveInteger(
    scenario.execution.maxEvents,
    "execution.maxEvents",
    add,
  );
  for (const [key, value] of Object.entries(
    scenario.execution.parallelism,
  )) {
    validatePositiveInteger(value, `execution.parallelism.${key}`, add);
  }

  const domains = indexUnique(
    scenario.memoryDomains,
    "memoryDomains",
    issues,
  );
  const devices = indexUnique(scenario.devices, "devices", issues);
  const links = indexUnique(scenario.links, "links", issues);
  indexUnique(scenario.placements, "placements", issues, "partitionId");
  indexUnique(scenario.transfers, "transfers", issues);
  const groups = indexUnique(scenario.groups, "groups", issues);

  for (const [index, domain] of scenario.memoryDomains.entries()) {
    const path = `memoryDomains[${index}]`;
    validatePositiveInteger(domain.capacityBytes, `${path}.capacityBytes`, add);
    validatePositiveInteger(
      domain.bandwidthBytesPerSec,
      `${path}.bandwidthBytesPerSec`,
      add,
    );
    validateNonNegativeInteger(domain.latencyNs, `${path}.latencyNs`, add);
    validateUniqueStrings(
      domain.allocationClasses,
      `${path}.allocationClasses`,
      add,
    );
    validateUniqueStrings(domain.accessibleBy, `${path}.accessibleBy`, add);
    if (domain.allocationClasses.length === 0) {
      add("empty_classes", `${path}.allocationClasses`, "must not be empty");
    }
    for (const deviceId of domain.accessibleBy) {
      if (!devices.has(deviceId)) {
        add(
          "unknown_device",
          `${path}.accessibleBy`,
          `unknown device ${deviceId}`,
        );
      }
    }
    validateGovernor(domain, devices, path, add);
    if (
      domain.kind === "unified"
      && domain.allocationClasses.includes("device")
    ) {
      add(
        "unified_class",
        `${path}.allocationClasses`,
        "unified domains must use the unified allocation class",
      );
    }
    if (domain.kind === "storage") {
      if (
        domain.allocationClasses.length !== 1
        || domain.allocationClasses[0] !== "storage"
      ) {
        add(
          "storage_class",
          `${path}.allocationClasses`,
          "storage domains must exclusively use the storage allocation class",
        );
      }
      if (domain.governor.kind !== "none") {
        add(
          "storage_governor",
          `${path}.governor`,
          "storage domains must not use host or device memory governors",
        );
      }
      if (domain.accessibleBy.length === 0) {
        add(
          "storage_access",
          `${path}.accessibleBy`,
          "storage domains require at least one local CPU endpoint",
        );
      }
      for (const deviceId of domain.accessibleBy) {
        const device = devices.get(deviceId);
        if (
          device !== undefined
          && (device.kind !== "cpu" || device.nodeId !== domain.nodeId)
        ) {
          add(
            "storage_access",
            `${path}.accessibleBy`,
            `${deviceId} must be a CPU on storage node ${domain.nodeId}`,
          );
        }
      }
    } else if (domain.allocationClasses.includes("storage")) {
      add(
        "storage_class",
        `${path}.allocationClasses`,
        "non-storage domains cannot use the storage allocation class",
      );
    }
  }

  for (const [index, device] of scenario.devices.entries()) {
    const path = `devices[${index}]`;
    if (device.id.length === 0 || device.nodeId.length === 0) {
      add("empty_id", path, "device id and nodeId must not be empty");
    }
    if (device.executionProvider.length === 0) {
      add("empty_ep", `${path}.executionProvider`, "must not be empty");
    }
    validatePositiveInteger(
      device.maxConcurrentCompute,
      `${path}.maxConcurrentCompute`,
      add,
    );
    validateUniqueStrings(
      device.memoryDomainIds,
      `${path}.memoryDomainIds`,
      add,
    );
    validateUniqueStrings(device.capabilities, `${path}.capabilities`, add);
    validateUniqueStrings(
      device.supportedDtypes,
      `${path}.supportedDtypes`,
      add,
    );
    if (device.memoryDomainIds.length === 0) {
      add("no_memory", `${path}.memoryDomainIds`, "must not be empty");
    }
    for (const domainId of device.memoryDomainIds) {
      const domain = domains.get(domainId);
      if (!domain) {
        add(
          "unknown_domain",
          `${path}.memoryDomainIds`,
          `unknown domain ${domainId}`,
        );
      } else if (!domain.accessibleBy.includes(device.id)) {
        add(
          "asymmetric_access",
          `${path}.memoryDomainIds`,
          `${domainId} does not grant access to ${device.id}`,
        );
      }
    }
  }

  for (const [index, link] of scenario.links.entries()) {
    const path = `links[${index}]`;
    if (!domains.has(link.sourceDomainId)) {
      add(
        "unknown_domain",
        `${path}.sourceDomainId`,
        `unknown domain ${link.sourceDomainId}`,
      );
    }
    if (!domains.has(link.targetDomainId)) {
      add(
        "unknown_domain",
        `${path}.targetDomainId`,
        `unknown domain ${link.targetDomainId}`,
      );
    }
    if (link.sourceDomainId === link.targetDomainId) {
      add("self_link", path, "link endpoints must differ");
    }
    validatePositiveInteger(
      link.bandwidthBytesPerSec,
      `${path}.bandwidthBytesPerSec`,
      add,
    );
    validateNonNegativeInteger(link.latencyNs, `${path}.latencyNs`, add);
    validatePositiveInteger(
      link.concurrencyLanes,
      `${path}.concurrencyLanes`,
      add,
    );
  }

  const allocationOwners = new Map<string, string>();
  const allocationDomains = new Map<string, string>();
  const allocationReservations = new Map<
    string,
    AllocationReservation
  >();
  const chargedBytes = new Map<string, number>();
  for (const [index, placement] of scenario.placements.entries()) {
    const path = `placements[${index}]`;
    const device = devices.get(placement.deviceId);
    if (!device) {
      add(
        "unknown_device",
        `${path}.deviceId`,
        `unknown device ${placement.deviceId}`,
      );
    } else {
      for (const capability of placement.requiredCapabilities) {
        if (!device.capabilities.includes(capability)) {
          add(
            "missing_capability",
            `${path}.requiredCapabilities`,
            `${placement.deviceId} lacks ${capability}`,
          );
        }
      }
    }
    validateUniqueStrings(
      placement.requiredCapabilities,
      `${path}.requiredCapabilities`,
      add,
    );

    for (const [allocationIndex, allocation] of placement.allocations.entries()) {
      const allocationPath = `${path}.allocations[${allocationIndex}]`;
      validatePositiveInteger(allocation.bytes, `${allocationPath}.bytes`, add);
      const previousOwner = allocationOwners.get(allocation.physicalAllocationId);
      if (allocation.physicalAllocationId.length === 0 || previousOwner) {
        add(
          "duplicate_allocation",
          `${allocationPath}.physicalAllocationId`,
          previousOwner
            ? `already owned by ${previousOwner}`
            : "must not be empty",
        );
      } else {
        allocationOwners.set(
          allocation.physicalAllocationId,
          placement.partitionId,
        );
        allocationDomains.set(
          allocation.physicalAllocationId,
          allocation.domainId,
        );
        allocationReservations.set(
          allocation.physicalAllocationId,
          allocation,
        );
      }

      const domain = domains.get(allocation.domainId);
      if (!domain) {
        add(
          "unknown_domain",
          `${allocationPath}.domainId`,
          `unknown domain ${allocation.domainId}`,
        );
        continue;
      }
      if (!domain.allocationClasses.includes(allocation.allocationClass)) {
        add(
          "unsupported_allocation_class",
          `${allocationPath}.allocationClass`,
          `${allocation.domainId} does not support ${allocation.allocationClass}`,
        );
      }
      if (device && !device.memoryDomainIds.includes(allocation.domainId)) {
        add(
          "inaccessible_allocation",
          allocationPath,
          `${placement.deviceId} cannot access ${allocation.domainId}`,
        );
      }
      const current = chargedBytes.get(allocation.domainId) ?? 0;
      const total = checkedAddForValidation(
        current,
        allocation.bytes,
        `${allocationPath}.bytes`,
        add,
      );
      if (total !== undefined) {
        chargedBytes.set(allocation.domainId, total);
      }
    }
  }

  for (const [index, placement] of scenario.placements.entries()) {
    const path = `placements[${index}].sharedAllocationIds`;
    const sharedAllocationIds = placement.sharedAllocationIds ?? [];
    validateUniqueStrings(sharedAllocationIds, path, add);
    const owned = new Set(
      placement.allocations.map((allocation) => allocation.physicalAllocationId),
    );
    const device = devices.get(placement.deviceId);
    for (const allocationId of sharedAllocationIds) {
      if (owned.has(allocationId)) {
        add(
          "self_alias",
          path,
          `${allocationId} is already owned by ${placement.partitionId}`,
        );
        continue;
      }
      const domainId = allocationDomains.get(allocationId);
      if (!domainId) {
        add(
          "unknown_allocation",
          path,
          `unknown shared allocation ${allocationId}`,
        );
      } else if (device && !device.memoryDomainIds.includes(domainId)) {
        add(
          "inaccessible_allocation",
          path,
          `${placement.deviceId} cannot access shared allocation ${allocationId} in ${domainId}`,
        );
      }
    }
  }

  for (const [domainId, bytes] of chargedBytes) {
    const domain = domains.get(domainId);
    if (domain && bytes > domain.capacityBytes) {
      add(
        "domain_over_capacity",
        `memoryDomains.${domainId}`,
        `reservations ${bytes} exceed capacity ${domain.capacityBytes}`,
      );
    }
  }

  for (const [index, transfer] of scenario.transfers.entries()) {
    validateTransfer(
      transfer,
      index,
      domains,
      scenario.links,
      allocationReservations,
      add,
    );
  }

  const rankOwners = new Map<string, string>();
  for (const [index, group] of scenario.groups.entries()) {
    const path = `groups[${index}]`;
    if (group.orderedRanks.length === 0) {
      add("empty_group", `${path}.orderedRanks`, "must not be empty");
    }
    const localRanks = new Set<string>();
    const localDevices = new Set<string>();
    for (const [rankIndex, rank] of group.orderedRanks.entries()) {
      const rankPath = `${path}.orderedRanks[${rankIndex}]`;
      if (rank.rankId.length === 0 || localRanks.has(rank.rankId)) {
        add("duplicate_rank", `${rankPath}.rankId`, `invalid rank ${rank.rankId}`);
      }
      localRanks.add(rank.rankId);
      const priorDevice = rankOwners.get(rank.rankId);
      if (priorDevice && priorDevice !== rank.deviceId) {
        add(
          "rank_remap",
          `${rankPath}.rankId`,
          `${rank.rankId} maps to both ${priorDevice} and ${rank.deviceId}`,
        );
      } else {
        rankOwners.set(rank.rankId, rank.deviceId);
      }
      if (!devices.has(rank.deviceId)) {
        add(
          "unknown_device",
          `${rankPath}.deviceId`,
          `unknown device ${rank.deviceId}`,
        );
      }
      if (localDevices.has(rank.deviceId)) {
        add(
          "duplicate_group_device",
          `${rankPath}.deviceId`,
          `${rank.deviceId} appears twice in ${group.id}`,
        );
      }
      localDevices.add(rank.deviceId);
    }
  }

  const rankCount = new Set(rankOwners.keys()).size;
  const parallelism = scenario.execution.parallelism;
  const requiredRanks = checkedProductForValidation(
    [
      parallelism.tensor,
      parallelism.pipeline,
      parallelism.expert,
      parallelism.data,
    ],
    "execution.parallelism",
    add,
  );
  if (requiredRanks !== undefined && requiredRanks > Math.max(rankCount, 1)) {
    add(
      "parallelism_participants",
      "execution.parallelism",
      `requires ${requiredRanks} ranks but only ${rankCount} are declared`,
    );
  }

  for (const [index, coefficient] of scenario.calibration.coefficients.entries()) {
    if (
      coefficient.id.length === 0
      || coefficient.unit.length === 0
      || !Number.isFinite(coefficient.value)
      || coefficient.value < 0
    ) {
      add(
        "invalid_calibration",
        `calibration.coefficients[${index}]`,
        "id/unit must be non-empty and value must be finite and non-negative",
      );
    }
  }

  // Access these maps so duplicate-id diagnostics cover every collection even
  // when no later object references them.
  void links;
  void groups;

  return { valid: issues.length === 0, issues };
}

export function findTransferPath(
  scenario: Pick<SimulationScenario, "memoryDomains" | "links">,
  transfer: TransferRequirement,
): readonly string[] | undefined {
  if (transfer.sourceDomainId === transfer.targetDomainId) {
    return [transfer.sourceDomainId];
  }

  const domains = new Map(
    scenario.memoryDomains.map((domain) => [domain.id, domain]),
  );
  if (!domains.has(transfer.sourceDomainId) || !domains.has(transfer.targetDomainId)) {
    return undefined;
  }

  const adjacency = new Map<string, SimLinkSpec[]>();
  for (const link of scenario.links) {
    const outgoing = adjacency.get(link.sourceDomainId) ?? [];
    outgoing.push(link);
    adjacency.set(link.sourceDomainId, outgoing);
  }
  for (const links of adjacency.values()) {
    links.sort((left, right) => left.id.localeCompare(right.id));
  }

  const queue: Array<{ path: string[]; hasPinnedIntermediate: boolean }> = [{
    path: [transfer.sourceDomainId],
    hasPinnedIntermediate: false,
  }];
  const visited = new Set<string>([`${transfer.sourceDomainId}|false`]);
  while (queue.length > 0) {
    const candidate = queue.shift();
    if (!candidate) {
      break;
    }
    const { path, hasPinnedIntermediate } = candidate;
    const current = path[path.length - 1];
    for (const link of adjacency.get(current) ?? []) {
      const currentIsIntermediate = current !== transfer.sourceDomainId;
      const nextHasPinnedIntermediate = hasPinnedIntermediate || (
        currentIsIntermediate
        && domains.get(current)?.allocationClasses.includes("pinned") === true
      );
      const visitKey = `${link.targetDomainId}|${nextHasPinnedIntermediate}`;
      if (visited.has(visitKey)) {
        continue;
      }
      const nextPath = [...path, link.targetDomainId];
      if (link.targetDomainId === transfer.targetDomainId) {
        if (
          !transfer.requiresPinnedStaging
          || nextHasPinnedIntermediate
        ) {
          return nextPath;
        }
      }
      visited.add(visitKey);
      queue.push({
        path: nextPath,
        hasPinnedIntermediate: nextHasPinnedIntermediate,
      });
    }
  }
  return undefined;
}

function validateTransfer(
  transfer: TransferRequirement,
  index: number,
  domains: ReadonlyMap<string, MemoryDomainSpec>,
  links: readonly SimLinkSpec[],
  allocations: ReadonlyMap<
    string,
    AllocationReservation
  >,
  add: (code: string, path: string, message: string) => void,
): void {
  const path = `transfers[${index}]`;
  validatePositiveInteger(transfer.bytes, `${path}.bytes`, add);
  if (!domains.has(transfer.sourceDomainId)) {
    add(
      "unknown_domain",
      `${path}.sourceDomainId`,
      `unknown domain ${transfer.sourceDomainId}`,
    );
  }
  if (!domains.has(transfer.targetDomainId)) {
    add(
      "unknown_domain",
      `${path}.targetDomainId`,
      `unknown domain ${transfer.targetDomainId}`,
    );
  }
  const transferPath = findTransferPath(
    { memoryDomains: [...domains.values()], links },
    transfer,
  );
  if (!transferPath) {
    add(
      "no_transfer_path",
      path,
      transfer.requiresPinnedStaging
        ? "no directed path with a pinned intermediate domain"
        : "no directed transfer path",
    );
    return;
  }

  validateUniqueStrings(
    transfer.stagingAllocationIds,
    `${path}.stagingAllocationIds`,
    add,
  );
  if (!transfer.requiresPinnedStaging && transfer.stagingAllocationIds.length > 0) {
    add(
      "unexpected_staging",
      `${path}.stagingAllocationIds`,
      "staging allocations were supplied for a transfer that does not require them",
    );
  }
  if (transfer.requiresPinnedStaging) {
    const intermediateDomains = new Set(transferPath.slice(1, -1));
    const coveredDomains = new Set<string>();
    for (const allocationId of transfer.stagingAllocationIds) {
      const allocation = allocations.get(allocationId);
      if (!allocation) {
        add(
          "unknown_allocation",
          `${path}.stagingAllocationIds`,
          `unknown staging allocation ${allocationId}`,
        );
        continue;
      }
      const domain = domains.get(allocation.domainId);
      if (
        allocation.purpose !== "staging"
        || allocation.allocationClass !== "pinned"
        || allocation.bytes < transfer.bytes
        || !intermediateDomains.has(allocation.domainId)
        || domain?.allocationClasses.includes("pinned") !== true
      ) {
        add(
          "invalid_staging",
          `${path}.stagingAllocationIds`,
          `${allocationId} is not a sufficiently large pinned staging allocation on the transfer path`,
        );
        continue;
      }
      coveredDomains.add(allocation.domainId);
    }
    for (const domainId of intermediateDomains) {
      if (
        domains.get(domainId)?.allocationClasses.includes("pinned") === true
        && !coveredDomains.has(domainId)
      ) {
        add(
          "missing_staging",
          `${path}.stagingAllocationIds`,
          `no staging allocation covers intermediate domain ${domainId}`,
        );
      }
    }
  }
}

function validateGovernor(
  domain: MemoryDomainSpec,
  devices: ReadonlyMap<string, { readonly id: string; readonly nodeId: string }>,
  path: string,
  add: (code: string, path: string, message: string) => void,
): void {
  if (domain.governor.kind === "host") {
    if (domain.governor.nodeId !== domain.nodeId) {
      add(
        "governor_node",
        `${path}.governor`,
        `host governor ${domain.governor.nodeId} does not own node ${domain.nodeId}`,
      );
    }
  } else if (domain.governor.kind === "device") {
    const device = devices.get(domain.governor.deviceId);
    if (!device || device.nodeId !== domain.nodeId) {
      add(
        "governor_device",
        `${path}.governor`,
        `unknown or cross-node owner ${domain.governor.deviceId}`,
      );
    }
  }
}

function indexUnique<T extends object>(
  values: readonly T[],
  path: string,
  issues: ScenarioValidationIssue[],
  idKey: keyof T = "id" as keyof T,
): Map<string, T> {
  const result = new Map<string, T>();
  values.forEach((value, index) => {
    const id = value[idKey];
    if (typeof id !== "string" || id.length === 0 || result.has(id)) {
      issues.push({
        code: "duplicate_id",
        path: `${path}[${index}].${String(idKey)}`,
        message: `id must be non-empty and unique; got ${String(id)}`,
      });
    } else {
      result.set(id, value);
    }
  });
  return result;
}

function validateUniqueStrings(
  values: readonly string[],
  path: string,
  add: (code: string, path: string, message: string) => void,
): void {
  if (new Set(values).size !== values.length || values.some((value) => value.length === 0)) {
    add("duplicate_value", path, "values must be non-empty and unique");
  }
}

function validatePositiveInteger(
  value: number,
  path: string,
  add: (code: string, path: string, message: string) => void,
): void {
  if (!Number.isSafeInteger(value) || value <= 0) {
    add("positive_integer", path, `must be a positive safe integer; got ${value}`);
  }
}

function validateNonNegativeInteger(
  value: number,
  path: string,
  add: (code: string, path: string, message: string) => void,
): void {
  if (!Number.isSafeInteger(value) || value < 0) {
    add(
      "non_negative_integer",
      path,
      `must be a non-negative safe integer; got ${value}`,
    );
  }
}

function checkedAddForValidation(
  left: number,
  right: number,
  path: string,
  add: (code: string, path: string, message: string) => void,
): number | undefined {
  const result = left + right;
  if (!Number.isSafeInteger(result)) {
    add("integer_overflow", path, "sum exceeds Number.MAX_SAFE_INTEGER");
    return undefined;
  }
  return result;
}

function checkedProductForValidation(
  values: readonly number[],
  path: string,
  add: (code: string, path: string, message: string) => void,
): number | undefined {
  let result = 1;
  for (const value of values) {
    result *= value;
    if (!Number.isSafeInteger(result)) {
      add("integer_overflow", path, "product exceeds Number.MAX_SAFE_INTEGER");
      return undefined;
    }
  }
  return result;
}

export function domainSupportsClass(
  domain: MemoryDomainSpec,
  allocationClass: AllocationClass,
): boolean {
  return domain.allocationClasses.includes(allocationClass);
}

export function calculateScenarioMemoryLedger(
  scenario: SimulationScenario,
): readonly ScenarioMemoryLedgerEntry[] {
  assertValidScenario(scenario);
  const reserved = new Map<string, number>();
  for (const placement of scenario.placements) {
    for (const allocation of placement.allocations) {
      const current = reserved.get(allocation.domainId) ?? 0;
      const total = current + allocation.bytes;
      if (!Number.isSafeInteger(total)) {
        throw new ScenarioValidationError([{
          code: "integer_overflow",
          path: `placements.${placement.partitionId}`,
          message: "reservation sum exceeds Number.MAX_SAFE_INTEGER",
        }]);
      }
      reserved.set(allocation.domainId, total);
    }
  }
  return scenario.memoryDomains
    .map((domain) => {
      const reservedBytes = reserved.get(domain.id) ?? 0;
      return {
        domainId: domain.id,
        capacityBytes: domain.capacityBytes,
        reservedBytes,
        freeBytes: domain.capacityBytes - reservedBytes,
      };
    })
    .sort((left, right) => left.domainId.localeCompare(right.domainId));
}
