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

    if (targetR2 === undefined || targetR2 === 1) { // If no target R2 or target is 1, create a perfect line
        updatedPoints = updatedPoints.map(point => {
            const absorbance = firstAbsorbance + slope * (point.concentration - firstPoint.concentration);
            return { ...point, absorbance: parseFloat(absorbance.toFixed(4)) };
        });
    } else { // If there is a target R2, adjust the middle points
        const linearAbsorbances = updatedPoints.map(p => firstAbsorbance + slope * (p.concentration - firstPoint.concentration));
        
        const yMean = linearAbsorbances.reduce((s, v) => s + v, 0) / linearAbsorbances.length;
        const totalSumOfSquaresSST = linearAbsorbances.reduce((s, v) => s + (v - yMean) ** 2, 0);
        
        const targetSSE = totalSumOfSquaresSST * (1 - targetR2);

        if (targetSSE < 0) {
            throw new Error("Cannot achieve target R² as it is too high for this data.");
        }
        
        const numMiddlePoints = updatedPoints.length - 2;
        if (numMiddlePoints <= 0) {
            throw new Error("Not enough middle points to adjust for R².");
        }

        const errorPerPoint = Math.sqrt(targetSSE / numMiddlePoints);
        
        let cumulativeNoise = 0;
        for (let i = 1; i < updatedPoints.length - 1; i++) {
            // Apply noise that trends, but randomly oscillates around the trend
            const randomFactor = (Math.random() - 0.5) * 0.5; // smaller random oscillation
            const trendFactor = (i - numMiddlePoints / 2) / (numMiddlePoints / 2); // create a trend
            
            let noise = errorPerPoint * (trendFactor + randomFactor);
            
            // To ensure the overall trend is maintained while meeting R2
            // We alternate adding and subtracting but keep the magnitude based on a trend
            const noiseDirection = (i % 2 === 0) ? 1 : -1;
            noise = noiseDirection * Math.abs(noise);


            let newAbsorbance = linearAbsorbances[i] + noise;

            // Ensure absorbance values are monotonically increasing (or decreasing if slope is negative)
            const prevAbsorbance = updatedPoints[i-1].absorbance;
            if (slope > 0 && newAbsorbance < prevAbsorbance) {
                newAbsorbance = prevAbsorbance + Math.random() * 0.001; // add a tiny bit to keep it increasing
            } else if (slope < 0 && newAbsorbance > prevAbsorbance) {
                newAbsorbance = prevAbsorbance - Math.random() * 0.001; // subtract a tiny bit
            }


            updatedPoints[i].absorbance = parseFloat(Math.max(0, newAbsorbance).toFixed(4));
        }

        // Final pass to ensure monotonicity after random adjustments
        for (let i = 1; i < updatedPoints.length - 1; i++) {
            const prev = updatedPoints[i-1].absorbance;
            const current = updatedPoints[i].absorbance;
            const next = linearAbsorbances[i+1]; // compare to ideal next to not drift too far
            
            if (slope > 0) {
                if (current < prev) updatedPoints[i].absorbance = prev + 0.0001;
                if (current > next && i + 1 < updatedPoints.length -1) updatedPoints[i].absorbance = (prev + next) / 2;
            } else {
                 if (current > prev) updatedPoints[i].absorbance = prev - 0.0001;
                 if (current < next && i + 1 < updatedPoints.length -1) updatedPoints[i].absorbance = (prev + next) / 2;
            }
            updatedPoints[i].absorbance = parseFloat(Math.max(0, updatedPoints[i].absorbance).toFixed(4))
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
