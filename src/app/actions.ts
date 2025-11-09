
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


function generateAbsorbanceValues(
    meanConcentration: number,
    standardDeviation: number,
    samplesPerGroup: number,
    standardCurveEquation: string
): { absorbanceValues: number[] } {
    const equationMatch = standardCurveEquation.match(/y = ([\d.-]+)x \+ ([\d.-]+)/);
    if (!equationMatch) {
        throw new Error("Invalid standard curve equation format.");
    }
    const m = parseFloat(equationMatch[1]);
    const c = parseFloat(equationMatch[2]);

    if (isNaN(m) || isNaN(c)) {
        throw new Error("Could not parse slope or intercept from the standard curve equation.");
    }

    const concentrationValues = [];
    for (let i = 0; i < samplesPerGroup; i++) {
        // Generate concentration values based on the group's stats
        const concentration = generateNormalRandom(meanConcentration, standardDeviation);
        concentrationValues.push(concentration);
    }

    const absorbanceValues = concentrationValues.map(conc => {
        // Use the standard curve to find the corresponding absorbance (y = mx + c)
        const absorbance = m * conc + c;
        // Ensure absorbance is not negative
        return Math.max(0, absorbance);
    });

    return { absorbanceValues };
}


// Pure statistical function for an independent two-sample t-test
function performTTest(group1: { mean: number; sd: number; samples: number; }, group2: { mean: number; sd: number; samples: number; }): { pValue: number } {
    const { mean: mean1, sd: sd1, samples: n1 } = group1;
    const { mean: mean2, sd: sd2, samples: n2 } = group2;

    if (n1 <= 1 || n2 <= 1) {
        return { pValue: NaN }; // Not enough data
    }
    
    // Calculate the t-statistic
    const pooledStdDev = Math.sqrt(((n1 - 1) * sd1 * sd1 + (n2 - 1) * sd2 * sd2) / (n1 + n2 - 2));
    if (pooledStdDev === 0) return { pValue: 1.0 }; // If there's no variance, the means are the same for p-value purposes
    
    const tStatistic = (mean1 - mean2) / (pooledStdDev * Math.sqrt(1/n1 + 1/n2));
    
    // Degrees of freedom
    const df = n1 + n2 - 2;

    if (df <= 0) return { pValue: NaN };

    // This is a simplified p-value calculation using a normal distribution approximation,
    // which is reasonable for df > 30 but less accurate for small df.
    // For a production-grade tool, a better approximation or a library would be needed.
    const absT = Math.abs(tStatistic);
    
    // Approximation of the standard normal CDF
    const normalCdf = (x: number) => {
        return 0.5 * (1 + Math.tanh(Math.sqrt(2 / Math.PI) * (x + 0.044715 * Math.pow(x, 3))));
    }

    // Two-tailed p-value
    const pValue = 2 * (1 - normalCdf(absT));


    return { pValue };
}


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
    
    if (isNaN(firstPoint.absorbance) || isNaN(lastPoint.absorbance)) {
       throw new Error("First and last absorbance values must be numbers.");
    }

    if (targetR2 !== undefined && (targetR2 > 1 || targetR2 < 0)) {
        throw new Error("Target R² must be between 0 and 1.");
    }
     let updatedPoints = points.map(p => ({...p}));

    const slope = (lastPoint.absorbance - firstPoint.absorbance) / (lastPoint.concentration - firstPoint.concentration);

    // Step 1: Calculate the ideal linear absorbance values for all points (the line of best fit)
    updatedPoints = updatedPoints.map(point => {
        const idealAbsorbance = firstPoint.absorbance + slope * (point.concentration - firstPoint.concentration);
        return { ...point, absorbance: idealAbsorbance };
    });

    // If no target R2 or target is 1, return the perfect line
    if (targetR2 === undefined || targetR2 >= 0.9999) { 
        return updatedPoints.map(p => ({...p, absorbance: parseFloat(p.absorbance.toFixed(4))}));
    }
    
    // Step 2: Calculate the required standard deviation of the residuals (errors)
    // to achieve the target R-squared.
    const yMean = updatedPoints.reduce((sum, p) => sum + p.absorbance, 0) / updatedPoints.length;
    const totalSumOfSquaresSST = updatedPoints.reduce((sum, p) => sum + Math.pow(p.absorbance - yMean, 2), 0);

    if (totalSumOfSquaresSST === 0) {
         throw new Error("Cannot adjust R² because all absorbance values are identical. There is no variance.");
    }

    const numMiddlePoints = updatedPoints.length - 2;
    if (numMiddlePoints <= 0) {
        throw new Error("Not enough middle points to adjust for R². You need at least 3 points total.");
    }

    // R^2 = 1 - (SSE / SST) => SSE = SST * (1 - R^2)
    const targetSSE = totalSumOfSquaresSST * (1 - targetR2);
    
    // The variance of the residuals is SSE / (n-2) for linear regression.
    // The standard deviation is the square root of the variance.
    // We use numMiddlePoints because we only add noise to them.
    const stdDevOfResiduals = Math.sqrt(targetSSE / numMiddlePoints);

    // Step 3: Add normally distributed noise to the middle points
    for (let i = 1; i < updatedPoints.length - 1; i++) {
        const idealAbsorbance = updatedPoints[i].absorbance;
        // Generate noise with a mean of 0 and the calculated standard deviation
        const noise = generateNormalRandom(0, stdDevOfResiduals);
        updatedPoints[i].absorbance = idealAbsorbance + noise;
    }
    
    // Step 4: Final pass to ensure monotonicity and format numbers
    for (let i = 1; i < updatedPoints.length; i++) {
        const prevAbsorbance = updatedPoints[i - 1].absorbance;
        // Enforce the trend (increasing or decreasing) dictated by the slope
        if (slope > 0 && updatedPoints[i].absorbance < prevAbsorbance) {
            updatedPoints[i].absorbance = prevAbsorbance + Math.random() * 0.001; // Add tiny positive jitter
        } else if (slope < 0 && updatedPoints[i].absorbance > prevAbsorbance) {
            updatedPoints[i].absorbance = prevAbsorbance - Math.random() * 0.001; // Add tiny negative jitter
        }
    }
    
    // Format all to 4 decimal places and ensure no negative values
    return updatedPoints.map(p => ({
        ...p,
        absorbance: parseFloat(Math.max(0, p.absorbance).toFixed(4))
    }));
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
      
      const result = generateAbsorbanceValues(
        group.mean,
        group.sd,
        group.samples,
        standardCurveEquation
      );

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

export type StatisticalTestRunner = {
    group1: { name: string; mean: number; sd: number; samples: number };
    group2: { name:string; mean: number; sd: number; samples: number };
    test: string;
}

export async function performStatisticalTest(
  values: StatisticalTestRunner
): Promise<StatisticalTestResult> {
    try {
        let result: { pValue: number };
        switch(values.test) {
            case 't-test':
                result = performTTest(values.group1, values.group2);
                break;
            default:
                throw new Error(`Unknown or unimplemented statistical test: ${values.test}`);
        }
        
        if (isNaN(result.pValue)) {
            throw new Error("Calculation resulted in NaN. Check input data, especially sample sizes and standard deviations.");
        }

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

    
