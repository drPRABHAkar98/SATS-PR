'use server';

import { z } from 'zod';
import { absorbanceValueTraceback } from '@/ai/flows/absorbance-value-traceback';
import { runStatisticalTest } from '@/ai/flows/statistical-analysis';
import type { StatisticalTestInput } from '@/ai/flows/statistical-analysis.schemas';
import { calculateLinearRegression } from '@/lib/analysis';
import { formSchema } from './schemas';
import type { AbsorbanceValueTracebackInput } from '@/ai/flows/absorbance-value-traceback';
import type { StandardPoint } from './schemas';


export type AnalysisResult = {
  standardCurve: {
    m: number;
    c: number;
    rSquare: number;
  };
  groupResults: {
    groupName: string;
    absorbanceValues: number[];
  }[];
};

export async function adjustRsquared(points: StandardPoint[], targetR2?: number): Promise<StandardPoint[]> {
    if (points.length < 3) {
      throw new Error("You need at least three points to adjust for a target R².");
    }
    const firstPoint = points[0];
    const lastPoint = points[points.length - 1];

    if (firstPoint.concentration === lastPoint.concentration) {
        throw new Error("First and last concentration values cannot be the same.");
    }
    
    const firstAbsorbance = firstPoint.absorbance;
    const lastAbsorbance = lastPoint.absorbance;

    if (isNaN(firstAbsorbance) || isNaN(lastAbsorbance)) {
       throw new Error("First and last absorbance values must be numbers.");
    }

    if (targetR2 !== undefined && (targetR2 > 1 || targetR2 < 0)) {
        throw new Error("Target R² must be between 0 and 1.");
    }

    const slope = (lastAbsorbance - firstAbsorbance) / (lastPoint.concentration - firstPoint.concentration);
    
    // Create a deep copy to avoid modifying the original points array directly
    let updatedPoints = points.map(p => ({...p}));

    if (!targetR2) { // If no target R2, create a perfect line
        updatedPoints = updatedPoints.map(point => {
            const absorbance = firstAbsorbance + slope * (point.concentration - firstPoint.concentration);
            return { ...point, absorbance: parseFloat(absorbance.toFixed(4)) };
        });
    } else { // If there is a target R2, adjust the middle points
        const concentrations = updatedPoints.map(p => p.concentration);
        const linearAbsorbances = updatedPoints.map(p => firstAbsorbance + slope * (p.concentration - firstPoint.concentration));
        
        // Calculate Sum of Squares Total (SST) based on the perfect linear fit
        const yMean = linearAbsorbances.reduce((s, v) => s + v, 0) / linearAbsorbances.length;
        const totalSumOfSquaresSST = linearAbsorbances.reduce((s, v) => s + (v - yMean) ** 2, 0);

        // Calculate the required Sum of Squared Errors (SSE) for the target R²
        const targetSSE = totalSumOfSquaresSST * (1 - targetR2);

        if (targetSSE < 0) {
            throw new Error("Cannot achieve target R² as it is too high for this data.");
        }
        
        const numMiddlePoints = updatedPoints.length - 2;
        if (numMiddlePoints <= 0) {
            throw new Error("Not enough middle points to adjust for R².");
        }

        // Distribute the error among the middle points
        const errorPerPoint = Math.sqrt(targetSSE / numMiddlePoints);
        
        for (let i = 1; i < updatedPoints.length - 1; i++) {
            // Apply alternating noise to keep the regression line relatively stable
            const noise = (i % 2 === 0 ? 1 : -1) * errorPerPoint * (Math.random() * 0.4 + 0.8); // Add some randomness
            const newAbsorbance = linearAbsorbances[i] + noise;
            updatedPoints[i].absorbance = parseFloat(Math.max(0, newAbsorbance).toFixed(4));
        }
    }
    
    // Return the full list of updated points
    return updatedPoints;
}

export async function runAnalysis(
  values: z.infer<typeof formSchema>
): Promise<AnalysisResult> {
  try {
    const { groups, standardCurve } = values;

    // 1. Standard Curve Calculation
    const points = standardCurve.map(p => ({ x: p.concentration, y: p.absorbance }));
    const regression = calculateLinearRegression(points);

    if (isNaN(regression.m) || isNaN(regression.c)) {
        throw new Error("Could not calculate standard curve. Please check your data points.");
    }

    const groupResults = [];
    const standardCurveEquation = `y = ${regression.m.toFixed(4)}x + ${regression.c.toFixed(4)}`;

    // 2. Individual Sample Absorbance Calculation for each group
    for (const group of groups) {
      const aiInput: AbsorbanceValueTracebackInput = {
        meanConcentration: group.mean,
        standardDeviation: group.sd,
        samplesPerGroup: group.samples,
        standardCurveEquation: standardCurveEquation,
      };

      const result = await absorbanceValueTraceback(aiInput);
      groupResults.push({
        groupName: group.name,
        absorbanceValues: result.absorbanceValues,
      });
    }

    return {
      standardCurve: {
        m: regression.m,
        c: regression.c,
        rSquare: regression.rSquare,
      },
      groupResults,
    };
  } catch (error) {
    console.error("Analysis failed:", error);
    if (error instanceof Error) {
        throw new Error(`Analysis failed: ${error.message}`);
    }
    throw new Error('An unknown error occurred during analysis.');
  }
}

export type StatisticalTestResult = {
  pValue: number;
};

export async function performStatisticalTest(
  values: StatisticalTestInput
): Promise<StatisticalTestResult> {
    try {
        const result = await runStatisticalTest(values);
        return {
            pValue: result.pValue,
        };
    } catch (error) {
        console.error("Statistical test failed:", error);
        if (error instanceof Error) {
            throw new Error(`Statistical test failed: ${error.message}`);
        }
        throw new Error('An unknown error occurred during the statistical test.');
    }
}
