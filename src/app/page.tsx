
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
  Sigma,
  Wand2,
  CheckCircle2,
  Sparkles,
  Download,
  BookUser,
} from "lucide-react";

import { cn } from "@/lib/utils";
import { Button } from "@/components/ui/button";
import {
  Form,
  FormControl,
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
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import { Separator } from "@/components/ui/separator";
import { useToast } from "@/hooks/use-toast";
import type { AnalysisResult, StatisticalTestResult } from "./actions";
import { runAnalysis, performStatisticalTest, adjustRsquared } from "./actions";
import { formSchema } from "./schemas";
import type { StatisticalTest, StandardPoint } from "./schemas";
import { calculateLinearRegression } from "@/lib/analysis";
import type { StatisticalTestInput } from "@/ai/flows/statistical-analysis.schemas";
import { Accordion, AccordionContent, AccordionItem, AccordionTrigger } from "@/components/ui/accordion";


// Helper function to calculate standard deviation
const calculateSD = (data: number[]): number => {
    const n = data.length;
    if (n < 2) return 0;
    const mean = data.reduce((a, b) => a + b) / n;
    const variance = data.reduce((a, b) => a + Math.pow(b - mean, 2), 0) / (n - 1);
    return Math.sqrt(variance);
};

export type TestResultState = {
  [key: string]: {
    isLoading: boolean;
    result: StatisticalTestResult | null;
  }
}

type StandardCurveInfo = {
    m: number;
    c: number;
    rSquare: number;
} | null;


export default function Home() {
  const { toast } = useToast();
  const [analysisResult, setAnalysisResult] = useState<AnalysisResult | null>(
    null
  );
  const [isLoading, setIsLoading] = useState(false);
  const [isAdjusting, setIsAdjusting] = useState(false);
  const [testResults, setTestResults] = useState<TestResultState>({});
  const [standardCurveInfo, setStandardCurveInfo] = useState<StandardCurveInfo>(null);


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
        { concentration: 0, absorbance: 0 },
        { concentration: 10, absorbance: 0.2 },
        { concentration: 20, absorbance: 0.4 },
        { concentration: 30, absorbance: 0.6 },
      ],
      statisticalTests: [
        {
          group1: "Normal",
          group2: "Diseased",
          test: "t-test",
          significanceLevel: "0.05",
        }
      ],
      targetR2: 0.99
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
    update: updateStandardPoint,
    replace: replaceStandardCurve
  } = useFieldArray({
    control: form.control,
    name: "standardCurve",
  });

  const {
    fields: statTestFields,
    append: appendStatTest,
    remove: removeStatTest,
  } = useFieldArray({
    control: form.control,
    name: "statisticalTests",
  });

  const watchedStandardCurve = form.watch('standardCurve');
  
  const updateCurveInfo = (data: {concentration: number, absorbance: number}[]) => {
      const points = data.map(p => ({ x: p.concentration, y: p.absorbance }));
      if (points.length >= 2) {
          const regression = calculateLinearRegression(points);
          if (!isNaN(regression.m) && !isNaN(regression.c)) {
              setStandardCurveInfo(regression);
          } else {
              setStandardCurveInfo(null);
          }
      } else {
          setStandardCurveInfo(null);
      }
  }

  useEffect(() => {
      updateCurveInfo(watchedStandardCurve);
  }, [watchedStandardCurve]);


  async function autoFillAbsorbance() {
    const points = form.getValues("standardCurve");
    const targetR2Value = form.getValues("targetR2");
    // Ensure targetR2 is either a number or undefined, not an empty string or other falsy values.
    const targetR2 = targetR2Value === "" || targetR2Value === null ? undefined : Number(targetR2Value);


    if (points.length < 2) {
      toast({
        variant: "destructive",
        title: "Not enough data points",
        description: "You need at least two points to create a linear curve.",
      });
      return;
    }
    
    setIsAdjusting(true);
    try {
        const adjustedPoints = await adjustRsquared(points, targetR2);
        replaceStandardCurve(adjustedPoints);
        updateCurveInfo(adjustedPoints);
        const finalRegression = calculateLinearRegression(adjustedPoints.map(p => ({x: p.concentration, y: p.absorbance})));

        toast({
            title: "Auto-fill Complete",
            description: `Absorbance values adjusted. New R² is ${finalRegression.rSquare.toFixed(4)}.`,
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
    try {
      const result = await runAnalysis(values);
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

  async function onRunTest(testIndex: number, testData: StatisticalTest) {
    const { group1: g1name, group2: g2name, test } = testData;
    const { groups } = form.getValues();

    if (!g1name || !g2name) {
      toast({ variant: "destructive", title: "Please select two groups to compare." });
      return;
    }

    if (g1name === g2name) {
      toast({ variant: "destructive", title: "Cannot compare a group to itself."});
      return;
    }

    const group1Data = groups.find(g => g.name === g1name);
    const group2Data = groups.find(g => g.name === g2name);
    
    if (!group1Data || !group2Data) {
      toast({ variant: "destructive", title: "Could not find selected groups."});
      return;
    }

    setTestResults(prev => ({ ...prev, [testIndex]: { isLoading: true, result: null }}));

    try {
      const testInput: StatisticalTestInput = {
        group1: {
          name: group1Data.name,
          mean: Number(group1Data.mean),
          sd: Number(group1Data.sd),
          samples: Number(group1Data.samples),
        },
        group2: {
          name: group2Data.name,
          mean: Number(group2Data.mean),
          sd: Number(group2Data.samples),
          samples: Number(group2Data.samples),
        },
        test: test,
      };
      const result = await performStatisticalTest(testInput);
      setTestResults(prev => ({ ...prev, [testIndex]: { isLoading: false, result }}));
      toast({ title: "Statistical test complete."});
    } catch (error) {
      console.error(error);
      toast({
        variant: "destructive",
        title: "Test Failed",
        description:
          error instanceof Error ? error.message : "An unknown error occurred.",
      });
      setTestResults(prev => ({ ...prev, [testIndex]: { isLoading: false, result: null }}));
    }
  }

  const forwardTestResults = React.useMemo(() => {
    if (!analysisResult) return null;

    const { m, c } = analysisResult.standardCurve;

    return analysisResult.groupResults.map(group => {
      const calculatedConcentrations = group.absorbanceValues.map(abs => (abs - c) / m);
      const concentrationMean = calculatedConcentrations.reduce((a, b) => a + b, 0) / calculatedConcentrations.length;
      const concentrationSD = calculateSD(calculatedConcentrations);

      return {
        groupName: group.groupName,
        sampleData: group.absorbanceValues.map((abs, i) => ({
            sample: i + 1,
            absorbance: abs,
            concentration: calculatedConcentrations[i],
        })),
        concentrationMean,
        concentrationSD,
      };
    });
  }, [analysisResult]);


  const watchedGroups = form.watch('groups');
  
  const handleExport = () => {
    if (!analysisResult || !forwardTestResults) return;
  
    const { analysisName, units, date, experimentName, standardCurve: standardCurveInputData, groups: initialGroups, statisticalTests } = form.getValues();
    const { m, c, rSquare } = analysisResult.standardCurve;
  
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
    csvData.push(["Equation", `y = ${m.toFixed(4)}x + ${c.toFixed(4)}`]);
    csvData.push(["R-squared", rSquare.toFixed(4)]);
    csvData.push([]); // Blank row
    csvData.push(["Standard Curve Data"]);
    csvData.push(["Concentration", "Absorbance"]);
    standardCurveInputData.forEach(p => {
      csvData.push([p.concentration, p.absorbance]);
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

    // 4. Statistical Test Results
    if (Object.keys(testResults).length > 0) {
      csvData.push(["Statistical Test Results"]);
      csvData.push(["Comparison", "Test Type", "Significance Level", "P-Value", "Result"]);
      statisticalTests.forEach((test, index) => {
        const testResult = testResults[index]?.result;
        if (testResult) {
          const significant = testResult.pValue < parseFloat(test.significanceLevel);
          csvData.push([
            `${test.group1} vs ${test.group2}`,
            test.test,
            `< ${test.significanceLevel}`,
            testResult.pValue.toExponential(4),
            significant ? "Significant" : "Not Significant"
          ]);
        }
      });
      csvData.push([]); // Blank row
    }
  
    // 5. Detailed Sample Data
    csvData.push(["Detailed Sample Data"]);
    csvData.push(["Group", "Sample", "TraceBack Absorbance", "Calculated Concentration"]);
    forwardTestResults.forEach(group => {
        group.sampleData.forEach(sample => {
            csvData.push([
                group.groupName,
                sample.sample,
                sample.absorbance.toFixed(4),
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
                    <Accordion type="multiple" className="w-full">
                      {groupFields.map((field, index) => {
                        const groupName = form.watch(`groups.${index}.name`);
                        return (
                          <AccordionItem value={`item-${index}`} key={field.id}>
                            <AccordionTrigger>
                              <div className="flex w-full items-center justify-between pr-4">
                                <span>Group: {groupName || `(Group ${index + 1})`}</span>
                                <Button
                                  type="button"
                                  variant="ghost"
                                  size="icon"
                                  onClick={(e) => {
                                    e.stopPropagation(); // prevent accordion from toggling
                                    removeGroup(index);
                                  }}
                                  disabled={groupFields.length <= 1}
                                  aria-label="Remove group"
                                  className="h-8 w-8"
                                >
                                  <Trash2 className="h-4 w-4 text-destructive" />
                                </Button>
                              </div>
                            </AccordionTrigger>
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

                {/* Statistical Analysis */}
                <Card>
                  <CardHeader>
                     <div className="flex items-center gap-3">
                      <div className="flex h-10 w-10 items-center justify-center rounded-lg bg-primary/10 text-primary">
                        <Sigma className="h-6 w-6" />
                      </div>
                      <div>
                        <CardTitle className="font-headline text-xl">
                          3. Statistical Analysis
                        </CardTitle>
                        <CardDescription>
                          Configure and run multiple statistical comparisons.
                        </CardDescription>
                      </div>
                    </div>
                  </CardHeader>
                  <CardContent className="space-y-2">
                     <Accordion type="multiple" className="w-full">
                       {statTestFields.map((field, index) => {
                        const testResult = testResults[index];
                        const currentTest = form.getValues('statisticalTests')[index];
                         return (
                          <AccordionItem value={`test-${index}`} key={field.id}>
                            <AccordionTrigger>
                              <div className="flex w-full items-center justify-between pr-4">
                                <span>Comparison #{index + 1}</span>
                                 <Button
                                  type="button"
                                  variant="ghost"
                                  size="icon"
                                  onClick={(e) => {
                                    e.stopPropagation();
                                    removeStatTest(index);
                                  }}
                                  disabled={statTestFields.length <= 1}
                                  aria-label="Remove test"
                                  className="h-8 w-8"
                                >
                                  <Trash2 className="h-4 w-4 text-destructive" />
                                </Button>
                              </div>
                            </AccordionTrigger>
                            <AccordionContent className="p-4 space-y-4">
                              <div className="grid grid-cols-1 gap-4 sm:grid-cols-2">
                                  <FormField
                                    control={form.control}
                                    name={`statisticalTests.${index}.group1`}
                                    render={({ field }) => (
                                      <FormItem>
                                        <FormLabel>Compare Group</FormLabel>
                                        <Select onValueChange={field.onChange} value={field.value}>
                                          <FormControl>
                                            <SelectTrigger>
                                              <SelectValue placeholder="Select a group" />
                                            </SelectTrigger>
                                          </FormControl>
                                          <SelectContent>
                                            {watchedGroups.map((g) => g.name && <SelectItem key={g.name} value={g.name}>{g.name}</SelectItem>)}
                                          </SelectContent>
                                        </Select>
                                        <FormMessage />
                                      </FormItem>
                                    )}
                                  />
                                  <FormField
                                    control={form.control}
                                    name={`statisticalTests.${index}.group2`}
                                    render={({ field }) => (
                                      <FormItem>
                                        <FormLabel>With Group</FormLabel>
                                        <Select onValueChange={field.onChange} value={field.value}>
                                          <FormControl>
                                            <SelectTrigger>
                                              <SelectValue placeholder="Select a group" />
                                            </SelectTrigger>
                                          </FormControl>
                                          <SelectContent>
                                            {watchedGroups.map((g) => g.name && <SelectItem key={g.name} value={g.name}>{g.name}</SelectItem>)}
                                          </SelectContent>
                                        </Select>
                                        <FormMessage />
                                      </FormItem>
                                    )}
                                  />
                                </div>
                              <div className="grid grid-cols-1 gap-4 sm:grid-cols-2">
                              <FormField
                                control={form.control}
                                name={`statisticalTests.${index}.test`}
                                render={({ field }) => (
                                  <FormItem>
                                    <FormLabel>Statistical Test</FormLabel>
                                    <Select
                                      onValueChange={field.onChange}
                                      defaultValue={field.value}
                                    >
                                      <FormControl>
                                        <SelectTrigger>
                                          <SelectValue placeholder="Select a test" />
                                        </SelectTrigger>
                                      </FormControl>
                                      <SelectContent>
                                        <SelectItem value="t-test">
                                          T-test
                                        </SelectItem>
                                        <SelectItem value="one-way-anova">
                                          One-way ANOVA
                                        </SelectItem>
                                        <SelectItem value="tukey-kramer">
                                          Tukey-Kramer test
                                        </SelectItem>
                                        <SelectItem value="mann-whitney">
                                          Mann-Whitney U
                                        </SelectItem>
                                        <SelectItem value="kruskal-wallis">
                                          Kruskal-Wallis
                                        </SelectItem>
                                      </SelectContent>
                                    </Select>
                                    <FormMessage />
                                  </FormItem>
                                )}
                              />
                              <FormField
                                control={form.control}
                                name={`statisticalTests.${index}.significanceLevel`}
                                render={({ field }) => (
                                  <FormItem>
                                    <FormLabel>Significance Level (p)</FormLabel>
                                    <Select
                                      onValueChange={field.onChange}
                                      defaultValue={field.value}
                                    >
                                      <FormControl>
                                        <SelectTrigger>
                                          <SelectValue placeholder="Select a p-value" />
                                        </SelectTrigger>
                                      </FormControl>
                                      <SelectContent>
                                        <SelectItem value="0.05">&lt; 0.05</SelectItem>
                                        <SelectItem value="0.01">&lt; 0.01</SelectItem>
                                        <SelectItem value="0.001">&lt; 0.001</SelectItem>
                                      </SelectContent>
                                    </Select>
                                    <FormMessage />
                                  </FormItem>
                                )}
                              />
                              </div>
                              <Button type="button" className="w-full" disabled={testResult?.isLoading} onClick={() => onRunTest(index, currentTest)}>
                                {testResult?.isLoading ? (
                                  <>
                                    <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                                    Calculating...
                                  </>
                                ) : (
                                  <>
                                  <Sparkles className="mr-2 h-5 w-5" />
                                  Run Test #{index + 1}
                                  </>
                                )}
                              </Button>
                              {testResult?.result && (
                                  <div className="space-y-2 rounded-lg border bg-muted/50 p-4">
                                      <h4 className="font-headline text-md font-semibold">Test Result</h4>
                                      <p className="text-sm">
                                          Calculated p-value: <span className="font-mono font-bold text-primary">{testResult.result.pValue.toExponential(4)}</span>
                                      </p>
                                      <p className={cn("text-sm font-medium", testResult.result.pValue < parseFloat(currentTest.significanceLevel) ? "text-green-500" : "text-amber-500")}>
                                          {testResult.result.pValue < parseFloat(currentTest.significanceLevel)
                                          ? "The difference is statistically significant."
                                          : "The difference is not statistically significant."}
                                      </p>
                                  </div>
                              )}
                            </AccordionContent>
                          </AccordionItem>
                         )
                        })}
                      </Accordion>
                      <Button
                        type="button"
                        variant="outline"
                        onClick={() =>
                          appendStatTest({
                            group1: '',
                            group2: '',
                            test: 't-test',
                            significanceLevel: '0.05',
                          })
                        }
                        className="w-full"
                      >
                        <Plus className="mr-2 h-4 w-4" /> Add Test
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
                          Provide data points to generate the standard curve.
                        </CardDescription>
                      </div>
                    </div>
                  </CardHeader>
                  <CardContent className="space-y-4">
                    <FormField
                        control={form.control}
                        name="targetR2"
                        render={({ field }) => (
                            <FormItem>
                            <FormLabel>Target R² (Optional)</FormLabel>
                            <FormControl>
                                <Input
                                type="number"
                                step="0.001"
                                min="0"
                                max="1"
                                placeholder="e.g., 0.995. Leave blank for perfect R²."
                                {...field}
                                onChange={(e) => field.onChange(e.target.value === '' ? '' : parseFloat(e.target.value))}
                                value={field.value ?? ''}
                                />
                            </FormControl>
                            <FormMessage />
                            </FormItem>
                        )}
                        />
                    <div className="max-h-60 space-y-2 overflow-y-auto pr-2">
                      {standardCurveFields.map((field, index) => (
                        <div
                          key={field.id}
                          className="grid grid-cols-[1fr_1fr_auto] items-end gap-2"
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
                                  <Input type="number" step="any" {...field} />
                                </FormControl>
                                <FormMessage />
                              </FormItem>
                            )}
                          />
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
                      ))}
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
                    {standardCurveInfo && (
                        <div className="mt-4 space-y-2 rounded-lg border bg-muted/50 p-4">
                            <h4 className="font-headline text-md font-semibold">
                                Curve Details
                            </h4>
                            <p className="text-sm font-medium">
                                Equation:{" "}
                                <span className="font-mono text-primary">{`y = ${standardCurveInfo.m.toFixed(
                                4
                                )}x + ${standardCurveInfo.c.toFixed(4)}`}</span>
                            </p>
                            <p className="text-sm font-medium">
                                R² Value:{" "}
                                <span className="font-mono text-primary">
                                {standardCurveInfo.rSquare.toFixed(4)}
                                </span>
                            </p>
                        </div>
                    )}
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
                      4. TraceBack Analysis Results
                    </CardTitle>
                    <CardDescription>
                      Review the calculated standard curve and traced-back absorbance values.
                    </CardDescription>
                  </div>
                </div>
              </CardHeader>
              <CardContent className="space-y-6">
                <div>
                  <h3 className="font-headline text-lg font-semibold">Standard Curve</h3>
                  <div className="mt-2 flex flex-wrap items-center gap-x-6 gap-y-2 rounded-lg border bg-muted/50 p-4">
                    <p className="text-sm font-medium">
                      Equation: <span className="font-mono text-primary">{`y = ${analysisResult.standardCurve.m.toFixed(4)}x + ${analysisResult.standardCurve.c.toFixed(4)}`}</span>
                    </p>
                    <p className="text-sm font-medium">
                      R² Value: <span className="font-mono text-primary">{analysisResult.standardCurve.rSquare.toFixed(4)}</span>
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
                              {group.absorbanceValues.map((_, index) => (
                                <TableHead key={index} className="text-center">Sample {index + 1}</TableHead>
                              ))}
                            </TableRow>
                          </TableHeader>
                          <TableBody>
                            <TableRow>
                              {group.absorbanceValues.map((value, index) => (
                                <TableCell key={index} className={cn("text-center font-mono")}>
                                  {value.toFixed(4)}
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
                          5. Forward Test Results (Validation)
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
                                            <TableHead className="text-center">TraceBack Absorbance</TableHead>
                                            <TableHead className="text-center">Recalculated Conc.</TableHead>
                                        </TableRow>
                                    </TableHeader>
                                    <TableBody>
                                        {group.sampleData.map(sample => (
                                            <TableRow key={sample.sample}>
                                                <TableCell className="text-center font-medium">{sample.sample}</TableCell>
                                                <TableCell className="text-center font-mono">{sample.absorbance.toFixed(4)}</TableCell>
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

    