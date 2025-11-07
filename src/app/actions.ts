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

    if (targetR2 && (targetR2 > 1 || targetR2 < 0)) {
        throw new Error("Target R² must be between 0 and 1.");
    }

    const slope = (lastAbsorbance - firstAbsorbance) / (lastPoint.concentration - firstPoint.concentration);
    let updatedPoints = points.map(p => ({...p}));

    if (!targetR2) {
        updatedPoints = updatedPoints.map(point => {
            const absorbance = firstAbsorbance + slope * (point.concentration - firstPoint.concentration);
            return { ...point, absorbance: parseFloat(absorbance.toFixed(4)) };
        });
    } else {
        const concentrations = updatedPoints.map(p => p.concentration);
        const linearAbsorbances = updatedPoints.map(p => firstAbsorbance + slope * (p.concentration - firstPoint.concentration));
        const yMean = linearAbsorbances.reduce((s, v) => s + v, 0) / linearAbsorbances.length;
        const totalSumOfSquaresSST = linearAbsorbances.reduce((s, v) => s + (v - yMean) ** 2, 0);
        const targetSSE = totalSumOfSquaresSST * (1 - targetR2);

        if (targetSSE < 0) {
            throw new Error("Cannot achieve target R² as it is too high for this data.");
        }

        let currentSSE = 0;
        let scaleFactor = 1;
        
        for (let iter = 0; iter < 50; iter++) {
            const tempPoints = updatedPoints.map(p => ({...p}));
            
            for (let i = 1; i < tempPoints.length - 1; i++) {
                const linearY = linearAbsorbances[i];
                const noise = (Math.sin(i * concentrations[i]) * 0.5); 
                const absorbanceRange = Math.abs(lastAbsorbance - firstAbsorbance);
                const error = noise * absorbanceRange * 0.1 * scaleFactor;
                tempPoints[i].absorbance = Math.max(0, linearY + error);
            }
            
            const currentRegression = calculateLinearRegression(tempPoints.map(p => ({x: p.concentration, y:p.absorbance})));
            const fittedValues = tempPoints.map(p => currentRegression.m * p.concentration + currentRegression.c);
            currentSSE = tempPoints.reduce((sum, p, i) => sum + (p.absorbance - fittedValues[i])**2, 0);

            if (Math.abs(currentSSE - targetSSE) < 1e-6) {
                 break; 
            }
            
            if(currentSSE > 0) {
                 scaleFactor *= Math.sqrt(targetSSE / currentSSE);
            } else {
                 scaleFactor = 1;
            }
        }
        
        for (let i = 1; i < updatedPoints.length - 1; i++) {
            const linearY = linearAbsorbances[i];
            const noise = (Math.sin(i * concentrations[i]) * 0.5);
            const absorbanceRange = Math.abs(lastAbsorbance - firstAbsorbance);
            const error = noise * absorbanceRange * 0.1 * scaleFactor;
            updatedPoints[i].absorbance = parseFloat(Math.max(0, linearY + error).toFixed(4));
        }
    }
    
    // Return only the updated points, not all of them
    const finalPoints = points.map((p, i) => {
        if (i > 0 && i < points.length - 1) {
            return updatedPoints[i];
        }
        return p;
    });

    return finalPoints;
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
