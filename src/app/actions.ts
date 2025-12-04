
'use server';

import { z } from 'zod';
import { calculateLinearRegression } from '@/lib/analysis';
import { formSchema } from './schemas';
import type { StandardPoint } from './schemas';


// A pure statistical function to generate normally distributed random numbers
// using the Box-Muller transform.
function generateNormalRandom(mean: number, stdDev: number): number {
    let u1, u2;
    do {
        u1 = Math.random();
        u2 = Math.random();
    } while (u1 === 0);

    const z = Math.sqrt(-2.0 * Math.log(u1)) * Math.cos(2.0 * Math.PI * u2);
    return z * stdDev + mean;
}

// Reverses the forward calculation steps to find the initial extrapolated concentration
function applyReverseSteps(finalConcentration: number, steps: z.infer<typeof formSchema>['forwardSteps']): number {
    // To reverse, we iterate through the steps backwards and apply the inverse operation
    return [...steps].reverse().reduce((currentValue, step) => {
        switch (step.operation) {
            case 'add':
                return currentValue - step.value;
            case 'subtract':
                return currentValue + step.value;
            case 'multiply':
                // Avoid division by zero if original value was 0
                return step.value !== 0 ? currentValue / step.value : currentValue;
            case 'divide':
                return currentValue * step.value;
            default:
                return currentValue;
        }
    }, finalConcentration);
}

function generateAbsorbanceValues(
    finalMeanConcentration: number,
    finalStandardDeviation: number,
    samplesPerGroup: number,
    slope: number,
    intercept: number,
    forwardSteps: z.infer<typeof formSchema>['forwardSteps']
): { absorbanceValues: number[], wellAbsorbances: { well: string, value: number }[][] } {

    if (isNaN(slope) || isNaN(intercept)) {
        throw new Error("Could not parse slope or intercept from the standard curve equation.");
    }

    // First, calculate the mean concentration *before* the forward steps were applied.
    const extrapolatedMeanConc = applyReverseSteps(finalMeanConcentration, forwardSteps);

    // Assumption: The standard deviation scales proportionally with the mean through the forward steps.
    // This is a simplification. A more rigorous approach would require more complex stats.
    // We calculate a scaling factor from the final mean to the extrapolated mean.
    const sdScalingFactor = finalMeanConcentration !== 0 ? extrapolatedMeanConc / finalMeanConcentration : 1;
    const extrapolatedSD = finalStandardDeviation * sdScalingFactor;

    const extrapolatedConcValues = [];
    for (let i = 0; i < samplesPerGroup; i++) {
        // Generate concentration values based on the *extrapolated* stats
        const concentration = generateNormalRandom(extrapolatedMeanConc, extrapolatedSD);
        extrapolatedConcValues.push(concentration);
    }

    const absorbanceValues = extrapolatedConcValues.map(conc => {
        // Use the standard curve to find the corresponding absorbance (y = mx + c)
        const absorbance = slope * conc + intercept;
        // Ensure absorbance is not negative
        return Math.max(0, absorbance);
    });

    const wellAbsorbances = absorbanceValues.map(meanAbs => {
        // Simulate small variations for 2 wells that average to the meanAbs
        const variation = (Math.random() - 0.5) * 0.02 * meanAbs; // up to 2% variation
        const well1 = meanAbs + variation;
        const well2 = meanAbs - variation;
        return [{ well: 'Well 1', value: Math.max(0, well1) }, { well: 'Well 2', value: Math.max(0, well2) }];
    });


    return { absorbanceValues, wellAbsorbances };
}

export type AnalysisResult = {
  standardCurve: {
    m: number;
    c: number;
  };
  groupResults: {
    groupName: string;
    absorbanceValues: number[];
    wellAbsorbances: { well: string; value: number; }[][];
    wellSelection: string[];
  }[];
};

export async function adjustRsquared(points: StandardPoint[], blankAbs: number, targetR2?: number): Promise<StandardPoint[]> {
    if (points.length < 3) {
      throw new Error("You need at least three points to adjust for a target R².");
    }
    
    // Use true absorbance (raw - blank) for calculations
    const truePoints = points.map(p => ({ ...p, absorbance: p.absorbance - blankAbs }));

    const firstPoint = truePoints[0];
    const lastPoint = truePoints[truePoints.length - 1];

    if (firstPoint.concentration === lastPoint.concentration) {
        throw new Error("First and last concentration values cannot be the same.");
    }
    
    if (isNaN(firstPoint.absorbance) || isNaN(lastPoint.absorbance)) {
       throw new Error("First and last absorbance values must be numbers. Ensure you have set them before auto-filling.");
    }

    if (targetR2 !== undefined && (targetR2 > 1 || targetR2 < 0)) {
        throw new Error("Target R² must be between 0 and 1.");
    }
    
    let updatedPoints = truePoints.map(p => ({...p}));
    const slope = (lastPoint.absorbance - firstPoint.absorbance) / (lastPoint.concentration - firstPoint.concentration);

    // Step 1: Calculate the ideal linear absorbance values for all points (the line of best fit)
    updatedPoints = updatedPoints.map(point => {
        const idealAbsorbance = firstPoint.absorbance + slope * (point.concentration - firstPoint.concentration);
        return { ...point, absorbance: idealAbsorbance };
    });

    // If no target R2 or target is 1, return the perfect line (plus the blank)
    if (targetR2 === undefined || targetR2 >= 0.9999) { 
        return updatedPoints.map(p => ({
            ...p, 
            absorbance: parseFloat((p.absorbance + blankAbs).toFixed(4))
        }));
    }
    
    // Step 2: Calculate the required standard deviation of the residuals (errors)
    const yMean = updatedPoints.reduce((sum, p) => sum + p.absorbance, 0) / updatedPoints.length;
    const totalSumOfSquaresSST = updatedPoints.reduce((sum, p) => sum + Math.pow(p.absorbance - yMean, 2), 0);

    if (totalSumOfSquaresSST === 0) {
         // This can happen if all points are already on a perfect horizontal line.
         // In this case, we can't introduce variance to meet a lower R2.
         return updatedPoints.map(p => ({...p, absorbance: parseFloat((p.absorbance + blankAbs).toFixed(4))}));
    }

    const numMiddlePoints = updatedPoints.length - 2;
    if (numMiddlePoints <= 0) {
        // Not enough points to add noise to.
        return updatedPoints.map(p => ({...p, absorbance: parseFloat((p.absorbance + blankAbs).toFixed(4))}));
    }

    // R^2 = 1 - (SSE / SST) => SSE = SST * (1 - R^2)
    const targetSSE = totalSumOfSquaresSST * (1 - targetR2);
    
    // The variance of the residuals is SSE / (n-2) for linear regression.
    // The standard deviation is the square root of the variance.
    const stdDevOfResiduals = Math.sqrt(targetSSE / numMiddlePoints);

    // Step 3: Add normally distributed noise to the middle points
    for (let i = 1; i < updatedPoints.length - 1; i++) {
        const idealAbsorbance = updatedPoints[i].absorbance;
        const noise = generateNormalRandom(0, stdDevOfResiduals);
        updatedPoints[i].absorbance = idealAbsorbance + noise;
    }
    
    // Step 4: Final pass to ensure monotonicity and format numbers
    for (let i = 1; i < updatedPoints.length; i++) {
        const prevAbsorbance = updatedPoints[i - 1].absorbance;
        if (slope > 0 && updatedPoints[i].absorbance < prevAbsorbance) {
            updatedPoints[i].absorbance = prevAbsorbance + Math.random() * 0.001; // Add tiny positive jitter
        } else if (slope < 0 && updatedPoints[i].absorbance > prevAbsorbance) {
            updatedPoints[i].absorbance = prevAbsorbance - Math.random() * 0.001; // Add tiny negative jitter
        }
    }
    
    // Format all to 4 decimal places, ensure no negative true OD, and add the blank back
    return updatedPoints.map(p => ({
        ...p,
        absorbance: parseFloat(Math.max(0, p.absorbance + blankAbs).toFixed(4))
    }));
}


export async function runAnalysis(
  values: z.infer<typeof formSchema>
): Promise<AnalysisResult> {
  try {
    const { groups, blankAbsorbance, slope, intercept, forwardSteps } = values;

    if (slope === undefined || intercept === undefined) {
        throw new Error("Slope and Intercept must be provided for the analysis.");
    }
    
    const groupResults = [];

    // 2. Individual Sample Absorbance Calculation for each group
    for (const group of groups) {
      
      const result = generateAbsorbanceValues(
        group.mean,
        group.sd,
        group.samples,
        slope,
        intercept,
        forwardSteps
      );

      // The generated absorbances are "true" values, so we add the blank back to simulate raw data.
      const rawAbsorbanceValues = result.absorbanceValues.map(abs => abs + blankAbsorbance);
      const rawWellAbsorbances = result.wellAbsorbances.map(wells => 
        wells.map(w => ({ ...w, value: w.value + blankAbsorbance }))
      );

      groupResults.push({
        groupName: group.name,
        absorbanceValues: rawAbsorbanceValues,
        wellAbsorbances: rawWellAbsorbances,
        wellSelection: group.wellSelection || []
      });
    }

    return {
      standardCurve: {
        m: slope,
        c: intercept,
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
