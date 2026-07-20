import * as SliderPrimitive from "@radix-ui/react-slider";
import { cn } from "../../lib/utils.js";

export function Slider({
  className,
  ...props
}: React.ComponentProps<typeof SliderPrimitive.Root>): React.JSX.Element {
  return (
    <SliderPrimitive.Root
      className={cn(
        "relative flex h-5 w-full touch-none select-none items-center",
        className,
      )}
      {...props}
    >
      <SliderPrimitive.Track className="relative h-1.5 w-full grow overflow-hidden rounded-full bg-zinc-200">
        <SliderPrimitive.Range className="absolute h-full bg-sky-700" />
      </SliderPrimitive.Track>
      <SliderPrimitive.Thumb className="block size-4 rounded-full border-2 border-sky-700 bg-white shadow-sm focus:outline-none focus:ring-2 focus:ring-sky-500" />
    </SliderPrimitive.Root>
  );
}
