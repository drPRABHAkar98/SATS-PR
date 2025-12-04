
import * as z from "zod";

export const groupSchema = z.object({
  name: z.string().min(1, "Name is required."),
  mean: z.coerce.number({ invalid_type_error: "Must be a number." }),
  sd: z.coerce.number({ invalid_type_error: "Must be a number." }).nonnegative("Cannot be negative."),
  samples: z.coerce
    .number({ invalid_type_error: "Must be a number." })
    .int()
    .min(1, "At least 1 sample."),
  usePlate: z.boolean().optional(),
  wellSelection: z.array(z.string()).optional(),
});

export const standardPointSchema = z.object({
  concentration: z.coerce.number({ invalid_type_error: "Must be a number." }).nonnegative(),
  absorbance: z.coerce.number({ invalid_type_error: "Must be a number." }).nonnegative(),
});

export const forwardStepSchema = z.object({
  operation: z.enum(['add', 'subtract', 'multiply', 'divide']),
  value: z.coerce.number({invalid_type_error: "Must be a number."}),
});

export type StandardPoint = z.infer<typeof standardPointSchema>;

export const formSchema = z.object({
  analysisName: z.string().optional(),
  units: z.string().optional(),
  date: z.string().optional(),
  experimentName: z.string().optional(),
  blankAbsorbance: z.coerce.number({invalid_type_error: "Must be a number"}).nonnegative("Cannot be negative"),
  targetR2: z.coerce.number().min(0).max(1).optional().nullable(),
  slope: z.coerce.number({invalid_type_error: "Slope must be a number"}).optional(),
  intercept: z.coerce.number({invalid_type_error: "Intercept must be a number"}).optional(),
  groups: z.array(groupSchema).min(1, "At least one group is required."),
  forwardSteps: z.array(forwardStepSchema),
  standardCurve: z
    .array(standardPointSchema)
    .min(2, "At least two points are needed for the curve."),
});
