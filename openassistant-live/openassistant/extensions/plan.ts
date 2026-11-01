import { Type } from "typebox";
import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";

// Tool results flow through the ordinary RPC stream; no extra transport is needed.
export default function (pi: ExtensionAPI) {
  pi.registerTool({
    name: "update_plan",
    label: "Update plan",
    description: "Publish or update a short task plan. Use at most one in_progress step.",
    parameters: Type.Object({
      steps: Type.Array(Type.Object({
        step: Type.String({ minLength: 1, maxLength: 240 }),
        status: Type.Union([
          Type.Literal("pending"), Type.Literal("in_progress"), Type.Literal("completed"),
        ]),
      }), { maxItems: 20 }),
    }),
    async execute(_id, { steps }) {
      if (steps.filter(step => step.status === "in_progress").length > 1) {
        throw new Error("Only one step can be in progress");
      }
      return { content: [{ type: "text", text: "Plan updated." }], details: { steps } };
    },
  });
}
