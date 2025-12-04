
"use client";

import * as React from "react";
import { zodResolver } from "@hookform/resolvers/zod";
import { useFieldArray, useForm } from "react-hook-form";
import * as z from "zod";
import { useState, useEffect } from "react";
import Papa from "papaparse";
import {
  FlaskConical,
  Plus,
  Trash2,
  Loader2,
  Info,
  LineChart,
  ClipboardList,
  FlaskRound,
  Calculator,
  Wand2,
  CheckCircle2,
  Download,
  BookUser,
  ChevronUp,
  ChevronDown,
} from "lucide-react";

import { cn } from "@/lib/utils";
import { Button } from "@/components/ui/button";
import {
  Form,
  FormControl,
  FormDescription,
  FormField,
  FormItem,
  FormLabel,
  FormMessage,
} from "@/components/ui/form";
import { Input } from "@/components/ui/input";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import { useToast } from "@/hooks/use-toast";
import type { AnalysisResult } from "./actions";
import { runAnalysis, adjustRsquared } from "./actions";
import { formSchema } from "./schemas";
import { calculateLinearRegression } from "@/lib/analysis";
import { Accordion, AccordionContent, AccordionItem, AccordionTrigger } from "@/components/ui/accordion";

// Helper function to calculate standard deviation
const calculateSD = (data: number[]): number => {
    const n = data.length;
    if (n < 2) return 0;
    const mean = data.reduce((a, b) => a + b) / n;
    const variance = data.reduce((a, b) => a + Math.pow(b - mean, 2), 0) / (n - 1);
    return Math.sqrt(variance);
};


export default function Home() {
  const { toast } = useToast();
  const [analysisResult, setAnalysisResult] = useState<AnalysisResult | null>(
    null
  );
  const [isLoading, setIsLoading] = useState(false);
  const [isAdjusting, setIsAdjusting] = useState(false);


  const form = useForm<z.infer<typeof formSchema>>({
    resolver: zodResolver(formSchema),
    defaultValues: {
      analysisName: "My Analysis",
      units: "ng/mL",
      date: new Date().toISOString().split("T")[0],
      experimentName: "Experiment 1",
      groups: [
        { name: "Normal", mean: 0, sd: 0, samples: 3 },
        { name: "Diseased", mean: 0, sd: 0, samples: 3 },
      ],
      standardCurve: [
        { concentration: 0, absorbance: 0.05 },
        { concentration: 10, absorbance: 0.2 },
        { concentration: 20, absorbance: 0.4 },
        { concentration: 30, absorbance: 0.6 },
        { concentration: 40, absorbance: 0.8 },
      ],
      blankAbsorbance: 0.05,
      targetR2: 0.995,
      slope: undefined,
      intercept: undefined,
    },
  });

  const {
    fields: groupFields,
    append: appendGroup,
    remove: removeGroup,
  } = useFieldArray({
    control: form.control,
    name: "groups",
  });

  const {
    fields: standardCurveFields,
    append: appendStandardPoint,
    remove: removeStandardPoint,
    replace: replaceStandardCurve
  } = useFieldArray({
    control: form.control,
    name: "standardCurve",
  });
  
  const watchedBlankAbsorbance = form.watch('blankAbsorbance');
  const watchedStandardCurve = form.watch('standardCurve');
  
  const [curveDetails, setCurveDetails] = useState({ m: 0, c: 0, rSquare: 0 });

  useEffect(() => {
    const validPoints = watchedStandardCurve.filter(
        (p) =>
        typeof p.concentration === 'number' &&
        !isNaN(p.concentration) &&
        typeof p.absorbance === 'number' &&
        !isNaN(p.absorbance)
    );

    if (validPoints.length >= 2) {
      const regression = calculateLinearRegression(
        validPoints.map(p => ({
          x: p.concentration,
          y: p.absorbance - (watchedBlankAbsorbance ?? 0),
        }))
      );
      setCurveDetails(regression);
    } else {
      setCurveDetails({ m: 0, c: 0, rSquare: 0 });
    }
  }, [watchedStandardCurve, watchedBlankAbsorbance]);

  async function autoFillAbsorbance() {
    const points = form.getValues("standardCurve");
    const targetR2 = form.getValues("targetR2");
    const blankAbsorbance = form.getValues("blankAbsorbance");

    if (points.length < 2) {
      toast({
        variant: "destructive",
        title: "Not enough data points",
        description: "You need at least two points to create a linear curve.",
      });
      return;
    }

    const firstPoint = points[0];
    const lastPoint = points[points.length - 1];

    if (firstPoint.concentration === lastPoint.concentration) {
      toast({
        variant: "destructive",
        title: "Invalid Concentration",
        description: "The first and last points of the curve cannot have the same concentration value.",
      });
      return;
    }
    
    setIsAdjusting(true);
    try {
        const adjustedPoints = await adjustRsquared(points, blankAbsorbance, targetR2);
        replaceStandardCurve(adjustedPoints);
        
        toast({
            title: "Auto-fill Complete",
            description: `Absorbance values adjusted. Plot these points in Excel to get the final equation.`,
        });

    } catch (error) {
        console.error(error);
        toast({
            variant: "destructive",
            title: "Auto-fill Failed",
            description: error instanceof Error ? error.message : "An unknown error occurred.",
        });
    } finally {
        setIsAdjusting(false);
    }
  }


  async function onSubmit(values: z.infer<typeof formSchema>) {
    setIsLoading(true);
    setAnalysisResult(null);

    // If slope/intercept are not provided, use the ones calculated from the curve
    const finalValues = { ...values };
    if (finalValues.slope === undefined || finalValues.intercept === undefined || isNaN(finalValues.slope) || isNaN(finalValues.intercept)) {
      if (curveDetails.m && curveDetails.c) {
        finalValues.slope = curveDetails.m;
        finalValues.intercept = curveDetails.c;
        toast({
          title: "Using App-Calculated Equation",
          description: "No manual equation provided. Using the equation calculated from the standard curve points.",
        });
      } else {
         toast({
          variant: "destructive",
          title: "Missing Curve Equation",
          description: "Please provide standard curve points to calculate an equation, or enter one manually.",
        });
        setIsLoading(false);
        return;
      }
    }


    try {
      const result = await runAnalysis(finalValues);
      setAnalysisResult(result);
      toast({
        title: "Analysis Complete",
        description: "The reverse calculation was successful.",
      });
    } catch (error) {
      console.error(error);
      toast({
        variant: "destructive",
        title: "Analysis Failed",
        description:
          error instanceof Error ? error.message : "An unknown error occurred.",
      });
    } finally {
      setIsLoading(false);
    }
  }

  const forwardTestResults = React.useMemo(() => {
    if (!analysisResult) return null;

    const { m, c } = analysisResult.standardCurve;
    const blankAbsorbance = form.getValues('blankAbsorbance');

    return analysisResult.groupResults.map(group => {
      // Use true absorbance (raw - blank) to calculate concentration
      const calculatedConcentrations = group.absorbanceValues.map(abs => (abs - blankAbsorbance - c) / m);
      const concentrationMean = calculatedConcentrations.reduce((a, b) => a + b, 0) / calculatedConcentrations.length;
      const concentrationSD = calculateSD(calculatedConcentrations);

      return {
        groupName: group.groupName,
        sampleData: group.absorbanceValues.map((abs, i) => ({
            sample: i + 1,
            rawAbsorbance: abs,
            trueAbsorbance: abs - blankAbsorbance,
            concentration: calculatedConcentrations[i],
        })),
        concentrationMean,
        concentrationSD,
      };
    });
  }, [analysisResult, form]);
  
  const handleExport = () => {
    if (!analysisResult || !forwardTestResults) return;
  
    const { analysisName, units, date, experimentName, standardCurve: standardCurveInputData, groups: initialGroups, blankAbsorbance } = form.getValues();
    const { m: slope, c: intercept } = analysisResult.standardCurve;
  
    let csvData: any[] = [];
  
    // 1. Analysis Details
    csvData.push(["Analysis Details"]);
    csvData.push(["Analysis Name", analysisName]);
    csvData.push(["Experiment Name", experimentName]);
    csvData.push(["Date", date]);
    csvData.push(["Concentration Units", units]);
    csvData.push([]); // Blank row
  
    // 2. Standard Curve Details
    csvData.push(["Standard Curve Details"]);
    csvData.push(["Equation Used for Analysis", `y = ${slope.toFixed(4)}x + ${intercept.toFixed(4)}`]);
    csvData.push(["Blank Absorbance", blankAbsorbance]);
    csvData.push([]); // Blank row
    csvData.push(["Standard Curve Raw Data (for reference)"]);
    csvData.push(["Concentration", "Raw Absorbance", "True Absorbance"]);
    standardCurveInputData.forEach(p => {
      csvData.push([p.concentration, p.absorbance, (p.absorbance - blankAbsorbance).toFixed(4)]);
    });
    csvData.push([]); // Blank row
  
    // 3. Group Summary
    csvData.push(["Group Summary"]);
    csvData.push(["Group Name", "Samples (n)", "Initial Mean Conc.", "Initial SD", "TraceBack Mean Conc.", "TraceBack SD"]);
    forwardTestResults.forEach(result => {
        const initialGroup = initialGroups.find(g => g.name === result.groupName);
        if (initialGroup) {
            csvData.push([
                result.groupName,
                initialGroup.samples,
                initialGroup.mean,
                initialGroup.sd,
                result.concentrationMean.toFixed(4),
                result.concentrationSD.toFixed(4)
            ]);
        }
    });
    csvData.push([]); // Blank row
  
    // 5. Detailed Sample Data
    csvData.push(["Detailed Sample Data"]);
    csvData.push(["Group", "Sample", "TraceBack Raw Absorbance", "Calculated Concentration"]);
    forwardTestResults.forEach(group => {
        group.sampleData.forEach(sample => {
            csvData.push([
                group.groupName,
                sample.sample,
                sample.rawAbsorbance.toFixed(4),
                sample.concentration.toFixed(4)
            ]);
        });
    });
  
    // Convert array of arrays to CSV string
    const csv = Papa.unparse(csvData);
  
    const blob = new Blob([csv], { type: 'text/csv;charset=utf-8;' });
    const link = document.createElement('a');
    const url = URL.createObjectURL(blob);
    link.setAttribute('href', url);
    link.setAttribute('download', `${analysisName || 'analysis'}-results.csv`);
    link.style.visibility = 'hidden';
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  
    toast({
      title: "Export Successful",
      description: "Your data has been downloaded as a CSV file.",
    });
  };

  const handleTargetR2Change = (increment: boolean) => {
    const currentValue = form.getValues('targetR2') ?? 0.99;
    const step = 0.005;
    let newValue = increment ? currentValue + step : currentValue - step;
    
    // Clamp the value between 0 and 1
    newValue = Math.max(0, Math.min(1, newValue));
    
    form.setValue('targetR2', parseFloat(newValue.toFixed(4)));
  };


  return (
    <div className="min-h-screen bg-background">
      <header className="sticky top-0 z-10 w-full border-b bg-card/80 backdrop-blur-sm">
        <div className="container mx-auto flex h-16 items-center gap-4 px-4 md:px-6">
          <FlaskConical className="h-8 w-8 text-primary" />
          <h1 className="font-headline text-xl font-bold tracking-tight text-foreground md:text-2xl">
            TraceBack Analytics <span className="text-sm font-normal text-muted-foreground">by prabha</span>
          </h1>
        </div>
      </header>

      <main className="container mx-auto p-4 md:p-6 lg:p-8">
        <Form {...form}>
          <form onSubmit={form.handleSubmit(onSubmit)} className="space-y-8">
            <Card>
              <CardHeader>
                <div className="flex items-center gap-3">
                  <div className="flex h-10 w-10 items-center justify-center rounded-lg bg-primary/10 text-primary">
                    <BookUser className="h-6 w-6" />
                  </div>
                  <div>
                    <CardTitle className="font-headline text-xl">
                      Analysis Details
                    </CardTitle>
                    <CardDescription>
                      Enter metadata for your analysis. This will be included in the export.
                    </CardDescription>
                  </div>
                </div>
              </CardHeader>
              <CardContent className="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-4">
                <FormField
                  control={form.control}
                  name="analysisName"
                  render={({ field }) => (
                    <FormItem>
                      <FormLabel>Analysis Name</FormLabel>
                      <FormControl>
                        <Input placeholder="e.g., ELISA Assay" {...field} />
                      </FormControl>
                      <FormMessage />
                    </FormItem>
                  )}
                />
                <FormField
                  control={form.control}
                  name="units"
                  render={({ field }) => (
                    <FormItem>
                      <FormLabel>Concentration Units</FormLabel>
                      <FormControl>
                        <Input placeholder="e.g., ng/mL" {...field} />
                      </FormControl>
                      <FormMessage />
                    </FormItem>
                  )}
                />
                <FormField
                  control={form.control}
                  name="date"
                  render={({ field }) => (
                    <FormItem>
                      <FormLabel>Date</FormLabel>
                      <FormControl>
                        <Input type="date" {...field} />
                      </FormControl>
                      <FormMessage />
                    </FormItem>
                  )}
                />
                <FormField
                  control={form.control}
                  name="experimentName"
                  render={({ field }) => (
                    <FormItem>
                      <FormLabel>Experiment Name</FormLabel>
                      <FormControl>
                        <Input placeholder="e.g., Exp 1" {...field} />
                      </FormControl>
                      <FormMessage />
                    </FormItem>
                  )}
                />
              </CardContent>
            </Card>

            <div className="grid grid-cols-1 gap-8 lg:grid-cols-2 lg:items-start">
              <div className="space-y-8">
                {/* Group Data */}
                <Card>
                  <CardHeader>
                    <div className="flex items-center gap-3">
                      <div className="flex h-10 w-10 items-center justify-center rounded-lg bg-primary/10 text-primary">
                        <ClipboardList className="h-6 w-6" />
                      </div>
                      <div>
                        <CardTitle className="font-headline text-xl">
                          1. Group Data
                        </CardTitle>
                        <CardDescription>
                          Input mean concentration, SD, and sample size for each group.
                        </CardDescription>
                      </div>
                    </div>
                  </CardHeader>
                  <CardContent className="space-y-2">
                    <Accordion type="multiple" className="w-full" defaultValue={groupFields.map((_, index) => `item-${index}`)}>
                      {groupFields.map((field, index) => {
                        const groupName = form.watch(`groups.${index}.name`);
                        return (
                          <AccordionItem value={`item-${index}`} key={field.id}>
                            <div className="flex items-center">
                              <AccordionTrigger className="flex-1 pr-2">
                                Group: {groupName || `(Group ${index + 1})`}
                              </AccordionTrigger>
                              <Button
                                type="button"
                                variant="ghost"
                                size="icon"
                                onClick={() => removeGroup(index)}
                                disabled={groupFields.length <= 1}
                                aria-label="Remove group"
                                className="h-8 w-8 shrink-0"
                              >
                                <Trash2 className="h-4 w-4 text-destructive" />
                              </Button>
                            </div>
                            <AccordionContent className="p-4">
                              <div className="grid grid-cols-1 gap-4">
                                <FormField
                                  control={form.control}
                                  name={`groups.${index}.name`}
                                  render={({ field }) => (
                                    <FormItem>
                                      <FormLabel>Group Name</FormLabel>
                                      <FormControl>
                                        <Input placeholder="e.g., Control" {...field} />
                                      </FormControl>
                                      <FormMessage />
                                    </FormItem>
                                  )}
                                />
                                <div className="grid grid-cols-3 gap-4">
                                  <FormField
                                    control={form.control}
                                    name={`groups.${index}.mean`}
                                    render={({ field }) => (
                                      <FormItem>
                                        <FormLabel>Mean Conc.</FormLabel>
                                        <FormControl>
                                          <Input type="number" step="any" {...field} />
                                        </FormControl>
                                        <FormMessage />
                                      </FormItem>
                                    )}
                                  />
                                  <FormField
                                    control={form.control}
                                    name={`groups.${index}.sd`}
                                    render={({ field }) => (
                                      <FormItem>
                                        <FormLabel>SD</FormLabel>
                                        <FormControl>
                                          <Input type="number" step="any" {...field} />
                                        </FormControl>
                                        <FormMessage />
                                      </FormItem>
                                    )}
                                  />
                                  <FormField
                                    control={form.control}
                                    name={`groups.${index}.samples`}
                                    render={({ field }) => (
                                      <FormItem>
                                        <FormLabel>Samples (n)</FormLabel>
                                        <FormControl>
                                          <Input type="number" {...field} />
                                        </FormControl>
                                        <FormMessage />
                                      </FormItem>
                                    )}
                                  />
                                </div>
                              </div>
                            </AccordionContent>
                          </AccordionItem>
                        )
                      })}
                    </Accordion>
                    <Button
                      type="button"
                      variant="outline"
                      onClick={() =>
                        appendGroup({ name: "", mean: 0, sd: 0, samples: 3 })
                      }
                      className="w-full"
                    >
                      <Plus className="mr-2 h-4 w-4" /> Add Group
                    </Button>
                  </CardContent>
                </Card>
              </div>

              <div className="space-y-8">
                {/* Standard Curve Data */}
                <Card>
                  <CardHeader>
                    <div className="flex items-center gap-3">
                      <div className="flex h-10 w-10 items-center justify-center rounded-lg bg-primary/10 text-primary">
                        <LineChart className="h-6 w-6" />
                      </div>
                      <div>
                        <CardTitle className="font-headline text-xl">
                          2. Standard Curve Data
                        </CardTitle>
                        <CardDescription>
                          This is sample data. Adjust points or use Auto-fill, then get the equation from Excel and enter it below.
                        </CardDescription>
                      </div>
                    </div>
                  </CardHeader>
                  <CardContent className="space-y-4">
                    <div className="grid grid-cols-1 gap-4">
                      <FormItem>
                          <FormLabel>Target R² (for Auto-fill)</FormLabel>
                          <div className="flex h-10 items-center justify-between rounded-md border border-input bg-background px-3">
                              <span className="font-mono text-sm">
                                  {(form.watch('targetR2') ?? 0).toFixed(4)}
                              </span>
                              <div className="flex flex-col">
                                  <Button
                                      type="button"
                                      variant="ghost"
                                      size="icon"
                                      className="h-4 w-4"
                                      onClick={() => handleTargetR2Change(true)}
                                      aria-label="Increase Target R-squared"
                                  >
                                      <ChevronUp className="h-3 w-3" />
                                  </Button>
                                  <Button
                                      type="button"
                                      variant="ghost"
                                      size="icon"
                                      className="h-4 w-4"
                                      onClick={() => handleTargetR2Change(false)}
                                      aria-label="Decrease Target R-squared"
                                  >
                                      <ChevronDown className="h-3 w-3" />
                                  </Button>
                              </div>
                          </div>
                      </FormItem>
                    </div>

                    <div className="max-h-60 space-y-2 overflow-y-auto pr-2">
                      {standardCurveFields.map((field, index) => {
                        const rawAbsorbance = watchedStandardCurve[index]?.absorbance ?? 0;
                        const trueAbsorbance = rawAbsorbance - (watchedBlankAbsorbance ?? 0);
                        return (
                        <div
                          key={field.id}
                          className="grid grid-cols-[1fr_1fr_1fr_auto] items-end gap-2"
                        >
                          <FormField
                            control={form.control}
                            name={`standardCurve.${index}.concentration`}
                            render={({ field }) => (
                              <FormItem>
                                {index === 0 && <FormLabel>Std. Conc.</FormLabel>}
                                <FormControl>
                                  <Input type="number" step="any" {...field} />
                                </FormControl>
                                <FormMessage />
                              </FormItem>
                            )}
                          />
                          <FormField
                            control={form.control}
                            name={`standardCurve.${index}.absorbance`}
                            render={({ field }) => (
                              <FormItem>
                                {index === 0 && <FormLabel>Absorbance</FormLabel>}
                                <FormControl>
                                  <Input type="number" step="any" {...field} value={isNaN(field.value) ? '' : field.value} />
                                </FormControl>
                                <FormMessage />
                              </FormItem>
                            )}
                          />
                          <FormItem>
                              {index === 0 && <FormLabel>True Abs.</FormLabel>}
                              <FormControl>
                                  <Input 
                                      type="number" 
                                      value={trueAbsorbance.toFixed(4)} 
                                      readOnly 
                                      disabled 
                                      className="bg-muted/70"
                                  />
                              </FormControl>
                          </FormItem>
                          <Button
                            type="button"
                            variant="ghost"
                            size="icon"
                            onClick={() => removeStandardPoint(index)}
                            disabled={standardCurveFields.length <= 2}
                            aria-label="Remove point"
                          >
                            <Trash2 className="h-4 w-4 text-destructive" />
                          </Button>
                        </div>
                      )})}
                    </div>
                     <div className="flex flex-col gap-2 sm:flex-row">
                      <Button
                        type="button"
                        variant="outline"
                        onClick={() =>
                          appendStandardPoint({ concentration: 0, absorbance: 0 })
                        }
                        className="flex-1"
                      >
                        <Plus className="mr-2 h-4 w-4" /> Add Point
                      </Button>
                      <Button
                        type="button"
                        variant="secondary"
                        onClick={autoFillAbsorbance}
                        className="flex-1"
                        disabled={standardCurveFields.length < 2 || isAdjusting}
                      >
                        {isAdjusting ? (
                          <>
                            <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                            Adjusting...
                          </>
                        ) : (
                          <>
                            <Wand2 className="mr-2 h-4 w-4" /> 
                            Auto-fill Absorbance
                          </>
                        )}
                      </Button>
                    </div>
                     <div className="mt-4 space-y-2 rounded-lg border bg-muted/50 p-4">
                        <h4 className="font-headline text-md font-semibold">
                            Curve Details (from True OD)
                        </h4>
                        <p className="text-sm">Equation: <span className="font-mono text-primary">{`y = ${curveDetails.m.toFixed(4)}x + ${curveDetails.c.toFixed(4)}`}</span></p>
                        <p className="text-sm">R² Value: <span className="font-mono text-primary">{curveDetails.rSquare.toFixed(4)}</span></p>
                    </div>
                     <div className="space-y-4 rounded-lg border bg-muted/50 p-4">
                        <h4 className="font-headline text-md font-semibold">
                            Manual Curve Equation (from Excel)
                        </h4>
                        <p className="text-xs text-muted-foreground">This equation will be used for the TraceBack analysis if provided.</p>
                        <div className="grid grid-cols-2 gap-4">
                           <FormField
                                control={form.control}
                                name="slope"
                                render={({ field }) => (
                                    <FormItem>
                                    <FormLabel>Slope (m)</FormLabel>
                                    <FormControl>
                                        <Input
                                        type="number"
                                        step="any"
                                        placeholder="e.g., 0.0188"
                                        {...field}
                                        />
                                    </FormControl>
                                    <FormMessage />
                                    </FormItem>
                                )}
                            />
                             <FormField
                                control={form.control}
                                name="intercept"
                                render={({ field }) => (
                                    <FormItem>
                                    <FormLabel>Y-Intercept (c)</FormLabel>
                                    <FormControl>
                                        <Input
                                        type="number"
                                        step="any"
                                        placeholder="e.g., 0.05"
                                        {...field}
                                        />
                                    </FormControl>
                                    <FormMessage />
                                    </FormItem>
                                )}
                            />
                        </div>
                        <FormField
                            control={form.control}
                            name="blankAbsorbance"
                            render={({ field }) => (
                                <FormItem>
                                <FormLabel>Blank Absorbance</FormLabel>
                                <FormControl>
                                    <Input
                                    type="number"
                                    step="any"
                                    placeholder="e.g., 0.05"
                                    {...field}
                                    />
                                </FormControl>
                                <FormMessage />
                                </FormItem>
                            )}
                        />
                    </div>
                  </CardContent>
                </Card>

                <Button type="submit" className="w-full" disabled={isLoading} size="lg">
                  {isLoading ? (
                    <>
                      <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                      Analyzing...
                    </>
                  ) : (
                    <>
                    <Calculator className="mr-2 h-5 w-5" />
                    Run TraceBack Analysis
                    </>
                  )}
                </Button>
              </div>
            </div>
          </form>
        </Form>

        {analysisResult && (
          <div className="mt-8 space-y-8">
            <Card>
              <CardHeader>
                <div className="flex items-center gap-3">
                  <div className="flex h-10 w-10 items-center justify-center rounded-lg bg-primary/10 text-primary">
                    <FlaskRound className="h-6 w-6" />
                  </div>
                  <div>
                    <CardTitle className="font-headline text-xl">
                      3. TraceBack Analysis Results
                    </CardTitle>
                    <CardDescription>
                      Review the calculated standard curve and traced-back absorbance values.
                    </CardDescription>
                  </div>
                </div>
              </CardHeader>
              <CardContent className="space-y-6">
                <div>
                  <h3 className="font-headline text-lg font-semibold">Standard Curve (from Manual Input)</h3>
                  <div className="mt-2 flex flex-wrap items-center gap-x-6 gap-y-2 rounded-lg border bg-muted/50 p-4">
                    <p className="text-sm font-medium">
                      Equation: <span className="font-mono text-primary">{`y = ${analysisResult.standardCurve.m.toFixed(4)}x + ${analysisResult.standardCurve.c.toFixed(4)}`}</span>
                    </p>
                  </div>
                </div>
                
                <div className="space-y-4">
                   <h3 className="font-headline text-lg font-semibold">TraceBack Absorbance Values</h3>
                  {analysisResult.groupResults.map((group) => (
                    <div key={group.groupName}>
                       <h4 className="font-semibold text-foreground">{group.groupName}</h4>
                      <div className="mt-2 overflow-x-auto rounded-lg border">
                        <Table>
                          <TableHeader>
                            <TableRow>
                              <TableHead className="w-32"></TableHead>
                              {group.absorbanceValues.map((_, index) => (
                                <TableHead key={index} className="text-center">Sample {index + 1}</TableHead>
                              ))}
                            </TableRow>
                          </TableHeader>
                          <TableBody>
                            <TableRow>
                              <TableCell className="font-medium">Raw Absorbance</TableCell>
                              {group.absorbanceValues.map((value, index) => (
                                <TableCell key={index} className="text-center font-mono">
                                  {value.toFixed(4)}
                                </TableCell>
                              ))}
                            </TableRow>
                             <TableRow>
                              <TableCell className="font-medium">True Absorbance</TableCell>
                              {group.absorbanceValues.map((value, index) => (
                                <TableCell key={index} className="text-center font-mono text-primary">
                                  {(value - form.getValues('blankAbsorbance')).toFixed(4)}
                                </TableCell>
                              ))}
                            </TableRow>
                          </TableBody>
                        </Table>
                      </div>
                    </div>
                  ))}
                   <div className="flex items-center gap-2 text-sm text-muted-foreground">
                      <Info className="h-4 w-4" />
                      <span>Values may fall outside a normal absorbance range.</span>
                  </div>
                </div>
              </CardContent>
            </Card>

            {forwardTestResults && (
              <Card>
                <CardHeader>
                  <div className="flex flex-wrap items-center justify-between gap-4">
                    <div className="flex items-center gap-3">
                      <div className="flex h-10 w-10 items-center justify-center rounded-lg bg-green-500/10 text-green-500">
                        <CheckCircle2 className="h-6 w-6" />
                      </div>
                      <div>
                        <CardTitle className="font-headline text-xl">
                          4. Forward Test Results (Validation)
                        </CardTitle>
                        <CardDescription>
                          Concentrations recalculated from absorbance values to verify the model.
                        </CardDescription>
                      </div>
                    </div>
                    <div className="flex gap-2">
                       <Button onClick={handleExport} variant="outline" size="sm">
                        <Download className="mr-2 h-4 w-4" />
                        Export to CSV
                      </Button>
                    </div>
                  </div>
                </CardHeader>
                <CardContent className="space-y-6">
                  {forwardTestResults.map((group) => {
                      const originalGroup = form.getValues('groups').find(g => g.name === group.groupName);
                      return (
                        <div key={group.groupName} className="space-y-4">
                            <h3 className="font-headline text-lg font-semibold text-foreground">{group.groupName}</h3>
                            <div className="grid grid-cols-1 gap-4 sm:grid-cols-2">
                                <div className="rounded-lg border bg-muted/50 p-4">
                                    <h4 className="text-sm font-medium text-muted-foreground">Original Input Data</h4>
                                    <p className="mt-1 text-2xl font-semibold">
                                        {Number(originalGroup?.mean).toFixed(2)} <span className="text-lg font-medium text-muted-foreground">± {Number(originalGroup?.sd).toFixed(2)}</span>
                                    </p>
                                    <p className="text-xs text-muted-foreground">Mean Conc. ± SD</p>
                                </div>
                                <div className="rounded-lg border bg-muted/50 p-4">
                                    <h4 className="text-sm font-medium text-muted-foreground">Forward Test Result</h4>
                                    <p className="mt-1 text-2xl font-semibold">
                                        {group.concentrationMean.toFixed(2)} <span className="text-lg font-medium text-muted-foreground">± {group.concentrationSD.toFixed(2)}</span>
                                    </p>
                                     <p className="text-xs text-muted-foreground">Recalculated Mean Conc. ± SD</p>
                                </div>
                            </div>
                            <div className="overflow-x-auto rounded-lg border">
                                <Table>
                                    <TableHeader>
                                        <TableRow>
                                            <TableHead className="text-center">Sample</TableHead>
                                            <TableHead className="text-center">Raw Absorbance</TableHead>
                                            <TableHead className="text-center">True Absorbance</TableHead>
                                            <TableHead className="text-center">Recalculated Conc.</TableHead>
                                        </TableRow>
                                    </TableHeader>
                                    <TableBody>
                                        {group.sampleData.map(sample => (
                                            <TableRow key={sample.sample}>
                                                <TableCell className="text-center font-medium">{sample.sample}</TableCell>
-                                                <TableCell className="text-center font-mono">{sample.absorbance.toFixed(4)}</TableCell>
+                                                <TableCell className="text-center font-mono">{sample.rawAbsorbance.toFixed(4)}</TableCell>
+                                                <TableCell className="text-center font-mono text-muted-foreground">{sample.trueAbsorbance.toFixed(4)}</TableCell>
                                                <TableCell className="text-center font-mono text-primary">{sample.concentration.toFixed(4)}</TableCell>
                                            </TableRow>
                                        ))}
                                    </TableBody>
                                </Table>
                            </div>
                        </div>
                      )
                  })}
                </CardContent>
              </Card>
            )}
          </div>
        )}
      </main>
    </div>
  );
}


    