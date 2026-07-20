import { useEffect, useState } from "react";
import {
  Check,
  Cpu,
  Link2,
  MemoryStick,
  Network,
} from "lucide-react";
import type {
  ComputeCapability,
  MemoryDomainSpec,
  SimDeviceSpec,
  SimLinkSpec,
  SimulationScenario,
} from "@inference-sim/core";
import { Badge } from "./components/ui/badge.js";
import { Button } from "./components/ui/button.js";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogTitle,
} from "./components/ui/dialog.js";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "./components/ui/select.js";
import { Switch } from "./components/ui/switch.js";
import {
  bytesToGibibytes,
  bytesToGigabytesPerSecond,
  COMPUTE_CAPABILITIES,
  finalizeEditedTopology,
  gibibytesToBytes,
  gigabytesPerSecondToBytes,
  LINK_KINDS,
} from "./topology-editor.js";
import TopologyGraph from "./TopologyGraph.js";

export default function TopologyEditorDialog({
  open,
  scenario,
  onOpenChange,
  onSave,
}: {
  readonly open: boolean;
  readonly scenario: SimulationScenario;
  readonly onOpenChange: (open: boolean) => void;
  readonly onSave: (scenario: SimulationScenario) => void;
}): React.JSX.Element {
  const [draft, setDraft] = useState(scenario);
  const [view, setView] = useState<"map" | "devices" | "links">("map");
  const [error, setError] = useState<string | undefined>();

  useEffect(() => {
    if (open) {
      setDraft(scenario);
      setView("map");
      setError(undefined);
    }
  }, [open, scenario]);

  const updateDevice = (
    id: string,
    update: (device: SimDeviceSpec) => SimDeviceSpec,
  ) => {
    setDraft((current) => ({
      ...current,
      devices: current.devices.map((device) => (
        device.id === id ? update(device) : device
      )),
    }));
  };
  const updateDomain = (
    id: string,
    update: (domain: MemoryDomainSpec) => MemoryDomainSpec,
  ) => {
    setDraft((current) => ({
      ...current,
      memoryDomains: current.memoryDomains.map((domain) => (
        domain.id === id ? update(domain) : domain
      )),
    }));
  };
  const updateLink = (
    id: string,
    update: (link: SimLinkSpec) => SimLinkSpec,
  ) => {
    setDraft((current) => ({
      ...current,
      links: current.links.map((link) => (
        link.id === id ? update(link) : link
      )),
    }));
  };
  const systemCount = new Set([
    ...draft.devices.map((device) => device.nodeId),
    ...draft.memoryDomains.map((domain) => domain.nodeId),
  ]).size;

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="flex w-[min(calc(100vw-2rem),960px)] max-h-[min(88vh,820px)] flex-col">
        <div className="border-b border-zinc-200 px-5 py-4 pr-14">
          <DialogTitle className="text-base font-bold">
            Device topology
          </DialogTitle>
          <DialogDescription className="mt-1 text-xs text-zinc-500">
            {draft.id} · {systemCount} system{systemCount === 1 ? "" : "s"} ·{" "}
            {draft.devices.length} compute chips · {draft.links.length} directed links
          </DialogDescription>
        </div>

        <div className="flex min-h-0 flex-1 flex-col">
          <div className="flex gap-1 border-b border-zinc-200 px-5 py-2">
            <Button
              type="button"
              variant={view === "map" ? "secondary" : "ghost"}
              onClick={() => setView("map")}
            >
              <Network className="size-4" />
              Map
            </Button>
            <Button
              type="button"
              variant={view === "devices" ? "secondary" : "ghost"}
              onClick={() => setView("devices")}
            >
              <Cpu className="size-4" />
              Devices
            </Button>
            <Button
              type="button"
              variant={view === "links" ? "secondary" : "ghost"}
              onClick={() => setView("links")}
            >
              <Link2 className="size-4" />
              Links
            </Button>
          </div>

          <div className="min-h-0 flex-1 overflow-y-auto bg-zinc-50 p-4 sm:p-5">
            {view === "map"
              ? (
                  <TopologyGraph
                    scenario={draft}
                    className="h-[min(65vh,620px)]"
                  />
                )
              : view === "devices"
              ? (
                  <div className="grid gap-3 xl:grid-cols-2">
                    {draft.devices.map((device) => (
                      <DeviceEditor
                        key={device.id}
                        device={device}
                        domains={device.memoryDomainIds.flatMap((id) => {
                          const domain = draft.memoryDomains.find(
                            (candidate) => candidate.id === id,
                          );
                          return domain === undefined ? [] : [domain];
                        })}
                        onDeviceChange={(update) => (
                          updateDevice(device.id, update)
                        )}
                        onDomainChange={updateDomain}
                      />
                    ))}
                  </div>
                )
              : (
                  <div className="grid gap-3 xl:grid-cols-2">
                    {draft.links.map((link) => (
                      <LinkEditor
                        key={link.id}
                        link={link}
                        domains={draft.memoryDomains}
                        onChange={(update) => updateLink(link.id, update)}
                      />
                    ))}
                  </div>
                )}
          </div>
        </div>

        <div className="border-t border-zinc-200 bg-white px-5 py-3">
          {error
            ? <div className="mb-2 text-xs text-rose-700">{error}</div>
            : null}
          <div className="flex justify-end gap-2">
            <Button
              type="button"
              variant="ghost"
              onClick={() => onOpenChange(false)}
            >
              Cancel
            </Button>
            <Button
              type="button"
              onClick={() => {
                try {
                  const edited = finalizeEditedTopology(draft);
                  onSave(edited);
                  onOpenChange(false);
                } catch (saveError) {
                  setError(
                    saveError instanceof Error
                      ? saveError.message
                      : String(saveError),
                  );
                }
              }}
            >
              <Check className="size-4" />
              Apply topology
            </Button>
          </div>
        </div>
      </DialogContent>
    </Dialog>
  );
}

function DeviceEditor({
  device,
  domains,
  onDeviceChange,
  onDomainChange,
}: {
  readonly device: SimDeviceSpec;
  readonly domains: readonly MemoryDomainSpec[];
  readonly onDeviceChange: (
    update: (device: SimDeviceSpec) => SimDeviceSpec,
  ) => void;
  readonly onDomainChange: (
    id: string,
    update: (domain: MemoryDomainSpec) => MemoryDomainSpec,
  ) => void;
}): React.JSX.Element {
  return (
    <section className="rounded-md border border-zinc-200 bg-white p-4">
      <div className="flex items-start justify-between gap-3">
        <div className="min-w-0">
          <h3 className="truncate text-sm font-bold">{device.id}</h3>
          <div className="truncate text-[11px] text-zinc-500">
            {device.nodeId}
          </div>
        </div>
        <Badge variant="neutral">{device.kind}</Badge>
      </div>

      <div className="mt-4 grid gap-3 sm:grid-cols-2">
        <EditorField label="Execution provider">
          <TextInput
            value={device.executionProvider}
            onChange={(value) => onDeviceChange((current) => ({
              ...current,
              executionProvider: value,
            }))}
          />
        </EditorField>
        <EditorField label="Compute lanes">
          <NumberInput
            value={device.maxConcurrentCompute}
            minimum={1}
            step={1}
            onChange={(value) => onDeviceChange((current) => ({
              ...current,
              maxConcurrentCompute: value,
            }))}
          />
        </EditorField>
        <EditorField label="Supported dtypes" className="sm:col-span-2">
          <TextInput
            value={device.supportedDtypes.join(", ")}
            onChange={(value) => onDeviceChange((current) => ({
              ...current,
              supportedDtypes: splitList(value),
            }))}
          />
        </EditorField>
      </div>

      <div className="mt-4 border-t border-zinc-200 pt-3">
        <div className="mb-2 text-xs font-semibold text-zinc-600">
          Capabilities
        </div>
        <div className="flex flex-wrap gap-2">
          {COMPUTE_CAPABILITIES.map((capability) => {
            const active = device.capabilities.includes(capability);
            return (
              <label
                key={capability}
                className="flex items-center gap-1.5 text-xs text-zinc-700"
              >
                <input
                  type="checkbox"
                  checked={active}
                  onChange={() => onDeviceChange((current) => ({
                    ...current,
                    capabilities: toggleCapability(
                      current.capabilities,
                      capability,
                    ),
                  }))}
                />
                {capability}
              </label>
            );
          })}
        </div>
      </div>

      {domains.map((domain) => (
        <div key={domain.id} className="mt-4 border-t border-zinc-200 pt-3">
          <div className="mb-3 flex items-center justify-between gap-2">
            <div className="flex min-w-0 items-center gap-2">
              <MemoryStick className="size-4 shrink-0 text-zinc-500" />
              <span className="truncate text-xs font-semibold">
                {domain.id}
              </span>
            </div>
            <Badge variant="neutral">{domain.kind}</Badge>
          </div>
          <div className="grid gap-3 sm:grid-cols-3">
            <EditorField label="Capacity GiB">
              <NumberInput
                value={bytesToGibibytes(domain.capacityBytes)}
                minimum={0.001}
                step={1}
                onChange={(value) => onDomainChange(domain.id, (current) => ({
                  ...current,
                  capacityBytes: gibibytesToBytes(value),
                }))}
              />
            </EditorField>
            <EditorField label="Bandwidth GB/s">
              <NumberInput
                value={bytesToGigabytesPerSecond(
                  domain.bandwidthBytesPerSec,
                )}
                minimum={0.001}
                step={1}
                onChange={(value) => onDomainChange(domain.id, (current) => ({
                  ...current,
                  bandwidthBytesPerSec:
                    gigabytesPerSecondToBytes(value),
                }))}
              />
            </EditorField>
            <EditorField label="Latency ns">
              <NumberInput
                value={domain.latencyNs}
                minimum={0}
                step={1}
                onChange={(value) => onDomainChange(domain.id, (current) => ({
                  ...current,
                  latencyNs: value,
                }))}
              />
            </EditorField>
          </div>
          <div className="mt-3 flex items-center justify-between">
            <span className="text-xs font-medium text-zinc-600">Coherent</span>
            <Switch
              checked={domain.coherent}
              onCheckedChange={(coherent) => (
                onDomainChange(domain.id, (current) => ({
                  ...current,
                  coherent,
                }))
              )}
            />
          </div>
        </div>
      ))}
    </section>
  );
}

function LinkEditor({
  link,
  domains,
  onChange,
}: {
  readonly link: SimLinkSpec;
  readonly domains: readonly MemoryDomainSpec[];
  readonly onChange: (update: (link: SimLinkSpec) => SimLinkSpec) => void;
}): React.JSX.Element {
  return (
    <section className="rounded-md border border-zinc-200 bg-white p-4">
      <div className="flex items-start justify-between gap-3">
        <div className="min-w-0">
          <h3 className="truncate text-sm font-bold">{link.id}</h3>
          <div className="truncate text-[11px] text-zinc-500">
            {link.sourceDomainId} → {link.targetDomainId}
          </div>
        </div>
        <Badge variant="neutral">{link.kind}</Badge>
      </div>
      <div className="mt-4 grid gap-3 sm:grid-cols-2">
        <EditorField label="Source">
          <DomainSelect
            value={link.sourceDomainId}
            domains={domains}
            onChange={(sourceDomainId) => onChange((current) => ({
              ...current,
              sourceDomainId,
            }))}
          />
        </EditorField>
        <EditorField label="Target">
          <DomainSelect
            value={link.targetDomainId}
            domains={domains}
            onChange={(targetDomainId) => onChange((current) => ({
              ...current,
              targetDomainId,
            }))}
          />
        </EditorField>
        <EditorField label="Link kind">
          <Select
            value={link.kind}
            onValueChange={(kind) => onChange((current) => ({
              ...current,
              kind: kind as SimLinkSpec["kind"],
            }))}
          >
            <SelectTrigger><SelectValue /></SelectTrigger>
            <SelectContent>
              {LINK_KINDS.map((kind) => (
                <SelectItem key={kind} value={kind}>{kind}</SelectItem>
              ))}
            </SelectContent>
          </Select>
        </EditorField>
        <EditorField label="Bandwidth GB/s">
          <NumberInput
            value={bytesToGigabytesPerSecond(link.bandwidthBytesPerSec)}
            minimum={0.001}
            step={1}
            onChange={(value) => onChange((current) => ({
              ...current,
              bandwidthBytesPerSec: gigabytesPerSecondToBytes(value),
            }))}
          />
        </EditorField>
        <EditorField label="Latency ns">
          <NumberInput
            value={link.latencyNs}
            minimum={0}
            step={1}
            onChange={(latencyNs) => onChange((current) => ({
              ...current,
              latencyNs,
            }))}
          />
        </EditorField>
        <EditorField label="Concurrency lanes">
          <NumberInput
            value={link.concurrencyLanes}
            minimum={1}
            step={1}
            onChange={(concurrencyLanes) => onChange((current) => ({
              ...current,
              concurrencyLanes,
            }))}
          />
        </EditorField>
      </div>
    </section>
  );
}

function DomainSelect({
  value,
  domains,
  onChange,
}: {
  readonly value: string;
  readonly domains: readonly MemoryDomainSpec[];
  readonly onChange: (value: string) => void;
}): React.JSX.Element {
  return (
    <Select value={value} onValueChange={onChange}>
      <SelectTrigger><SelectValue /></SelectTrigger>
      <SelectContent>
        {domains.map((domain) => (
          <SelectItem key={domain.id} value={domain.id}>
            {domain.id}
          </SelectItem>
        ))}
      </SelectContent>
    </Select>
  );
}

function EditorField({
  label,
  className = "",
  children,
}: {
  readonly label: string;
  readonly className?: string;
  readonly children: React.ReactNode;
}): React.JSX.Element {
  return (
    <label className={`block ${className}`}>
      <span className="mb-1.5 block text-[11px] font-semibold text-zinc-600">
        {label}
      </span>
      {children}
    </label>
  );
}

function TextInput({
  value,
  onChange,
}: {
  readonly value: string;
  readonly onChange: (value: string) => void;
}): React.JSX.Element {
  return (
    <input
      type="text"
      value={value}
      onChange={(event) => onChange(event.currentTarget.value)}
      className="h-9 w-full rounded-md border border-zinc-300 bg-white px-3 text-sm outline-none focus:ring-2 focus:ring-sky-500"
    />
  );
}

function NumberInput({
  value,
  minimum,
  step,
  onChange,
}: {
  readonly value: number;
  readonly minimum: number;
  readonly step: number;
  readonly onChange: (value: number) => void;
}): React.JSX.Element {
  return (
    <input
      type="number"
      value={value}
      min={minimum}
      step={step}
      onChange={(event) => {
        const next = Number(event.currentTarget.value);
        if (Number.isFinite(next)) {
          onChange(next);
        }
      }}
      className="h-9 w-full rounded-md border border-zinc-300 bg-white px-3 text-sm tabular-nums outline-none focus:ring-2 focus:ring-sky-500"
    />
  );
}

function splitList(value: string): string[] {
  return [...new Set(
    value.split(",").map((entry) => entry.trim()).filter(Boolean),
  )];
}

function toggleCapability(
  current: readonly ComputeCapability[],
  capability: ComputeCapability,
): ComputeCapability[] {
  return current.includes(capability)
    ? current.filter((item) => item !== capability)
    : [...current, capability].sort();
}
