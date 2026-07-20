import * as DialogPrimitive from "@radix-ui/react-dialog";
import { X } from "lucide-react";
import { cn } from "../../lib/utils.js";

export const Dialog = DialogPrimitive.Root;
export const DialogTrigger = DialogPrimitive.Trigger;
export const DialogClose = DialogPrimitive.Close;
export const DialogTitle = DialogPrimitive.Title;
export const DialogDescription = DialogPrimitive.Description;

export function DialogContent({
  className,
  children,
  ...props
}: React.ComponentProps<typeof DialogPrimitive.Content>): React.JSX.Element {
  return (
    <DialogPrimitive.Portal>
      <DialogPrimitive.Overlay className="fixed inset-0 z-50 bg-zinc-950/35" />
      <DialogPrimitive.Content
        className={cn(
          "fixed left-1/2 top-1/2 z-50 max-h-[min(80vh,720px)] w-[min(calc(100vw-2rem),720px)] -translate-x-1/2 -translate-y-1/2 overflow-hidden rounded-md border border-zinc-200 bg-white shadow-xl focus:outline-none",
          className,
        )}
        {...props}
      >
        {children}
        <DialogPrimitive.Close
          className="absolute right-3 top-3 grid size-8 place-items-center rounded-md text-zinc-500 hover:bg-zinc-100 focus:outline-none focus:ring-2 focus:ring-sky-500"
          aria-label="Close"
        >
          <X className="size-4" />
        </DialogPrimitive.Close>
      </DialogPrimitive.Content>
    </DialogPrimitive.Portal>
  );
}
