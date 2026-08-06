/*******************************************************************************
 * SolvePOMDP
 * Copyright (C) 2017 Erwin Walraven
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 *******************************************************************************/

package solver;

import java.io.File;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Random;

import pomdp.POMDP;
import pomdp.SolverProperties;
import solver.AlphaVector;
import solver.BeliefPoint;
import solver.OutputFileWriter;
import solver.ProbabilitySample;

/**
 * Solving POMDPs using point-based value iteration
 */

public class ERPerseus implements Solver {	
	private Random rnd;
	private SolverProperties sp;
	private long totalSolveTime = 0;
	private double expectedValue;
	private double lambda = 1.0;  // temperature parameter for entropy regularization (default: deterministic)
	
	public ERPerseus(SolverProperties solverProperties, Random rnd, double lambda) {
		this.rnd = rnd;
		this.sp = solverProperties;
		this.lambda = lambda;
	}
	
	/**
	 * Get lambda (temperature parameter)
	 * @return lambda value
	 */
	public double getLambda() {
		return lambda;
	}
	
	/**
	 * Set lambda (temperature parameter)
	 * @param lambda temperature parameter (0 = deterministic, higher = more stochastic)
	 */
	public void setLambda(double lambda) {
		this.lambda = lambda;
	}
	
	public String getType() {
		if (lambda > 0.0) {
			return "approximate (entropy-regularized perseus)";
		}
		return "approximate (perseus)";
	}

	public double getTotalSolveTime() {
		return totalSolveTime * 0.001;
	}
	
	/**
	 * Generates belief points by sampling reachable beliefs through simulated interaction
	 * with the POMDP. The generated set consists of the initial belief, beliefs sampled
	 * via random trajectories (using a random policy), and eventually corner beliefs.
	 *
	 * @param pomdp the POMDP environment
	 * @return an ArrayList of sampled BeliefPoints representing the reachable belief space
	 */
	private ArrayList<BeliefPoint> getBeliefPoints(POMDP pomdp) {
		// List of unique belief points collected during sampling
		ArrayList<BeliefPoint> B = new ArrayList<BeliefPoint>();
		HashSet<BeliefPoint> Bset = new HashSet<BeliefPoint>(); // Fast lookup for duplicates

		// Always include the initial belief
		B.add(pomdp.getInitialBelief());
		Bset.add(pomdp.getInitialBelief());

		// For the number of required sampling runs
		for (int run = 0; run < sp.getBeliefSamplingRuns(); run++) {
			BeliefPoint b = pomdp.getInitialBelief(); // Start each run from initial belief

			// Step forward through the POMDP for a given number of timesteps
			for (int step = 0; step < sp.getBeliefSamplingSteps(); step++) {
				pomdp.prepareBelief(b); // Prepare/correct the belief if necessary (e.g. normalization)

				// Sample action and observation randomly to generate successor belief

				// Select a random action
				int action = rnd.nextInt(pomdp.getNumActions());

				// Construct the probability mass over observations given (belief, action)
				ProbabilitySample ps = new ProbabilitySample(rnd);
				for (int obs = 0; obs < pomdp.getNumObservations(); obs++) {
					double prob = b.getActionObservationProbability(action, obs); // Probability for observation obs
					if (prob > 1.0) prob = 1.0; // Clamp probability to [0,1]
					ps.addItem(obs, prob); // Add this observation with its probability
				}

				// Sample an observation according to the probabilities
				int observation = ps.sampleItem();

				// Generate new belief point resulting from (belief, action, observation)
				BeliefPoint bao = pomdp.updateBelief(b, action, observation);

				// Update bao's history to reflect the step taken
				bao.setHistory(b.getHistoryCopy());       // Copy full action-observation history so far
				bao.addToHistory(action);                 // Add the taken action to history
				bao.addToHistory(observation);            // Add the observed outcome to history

				// Add the new belief point to B if it is novel (not already in Bset)
				if (!Bset.contains(bao)) {
					B.add(bao);
					Bset.add(bao);
				}

				// Continue simulation from this new belief
				b = bao;
			}
		}
		
		// add corner beliefs
		for(int s = 0; s < pomdp.getNumStates(); s++) {
			double[] beliefEntries = new double[pomdp.getNumStates()];
			beliefEntries[s] = 1.0;
			B.add(new BeliefPoint(beliefEntries));
		}
		
		return B;
	}
	
	/**
	 * Performs a single Perseus backup stage for entropy-regularized value iteration.
	 * 
	 * @param pomdp             The POMDP model object.
	 * @param immediateRewards  The list of immediate reward alpha vectors (one per action).
	 * @param V                 The current value function (list of alpha vectors).
	 * @param B                 The current set of belief points sampled from the belief space.
	 * @return                 The updated value function (list of alpha vectors) after this backup stage.
	 *
	 * The backup proceeds in the following steps:
	 *   1. For each value function vector, precompute all "gkao" alpha vectors needed for backups.
		*      These encode the one-step lookahead value for each (value vector k, action a, observation o) triple.
	 *   2. Maintain a working set Btilde of beliefs not yet improved by the current value backup.
	 *   3. While Btilde is non-empty:
	 *       - Randomly pick a belief b from Btilde.
	 *       - Compute the backup alpha vector for b (using softmax or hardmax, depending on lambda).
	 *       - If this new vector improves the value at b, add it to the next value set (Vnext);
	 *         otherwise, add the best existing vector from V at b.
	 *       - Recompute Btilde as those beliefs in B that are not yet improved by Vnext versus V.
	 *   4. Return Vnext as the new value function set.
	 */
	private ArrayList<AlphaVector> backupStage(
		POMDP pomdp, 
		ArrayList<AlphaVector> immediateRewards, 
		ArrayList<AlphaVector> V, 
		ArrayList<BeliefPoint> B
	) {
		int nStates = pomdp.getNumStates();
		int nActions = pomdp.getNumActions();
		int nObservations = pomdp.getNumObservations();
		
		ArrayList<AlphaVector> Vnext = new ArrayList<AlphaVector>();
		List<BeliefPoint> Btilde = new ArrayList<BeliefPoint>(B);  // Working set of beliefs not yet improved
		
		// === Step 1: Precompute all "gkao" (V,action,observation) backup alpha vectors ===
		// gkao[k][a][o] encodes the backup vector for value vector k, action a, and observation o
		AlphaVector[][][] gkao = new AlphaVector[V.size()][nActions][nObservations];
		for (int k = 0; k < V.size(); k++) {
			for (int a = 0; a < nActions; a++) {
				for (int o = 0; o < nObservations; o++) {
					double[] entries = new double[nStates];
					
					// Compute each entry for state s for this (k,a,o) triple
					for (int s = 0; s < nStates; s++) {
						double val = 0.0;
						// Sum over next states sPrime: observation prob * transition prob * value
						for (int sPrime = 0; sPrime < nStates; sPrime++) {
							val += pomdp.getObservationProbability(a, sPrime, o) *
							       pomdp.getTransitionProbability(s, a, sPrime) *
							       V.get(k).getEntry(sPrime);
						}
						entries[s] = val;
					}
					AlphaVector av = new AlphaVector(entries);
					av.setAction(a);
					gkao[k][a][o] = av;
				}
			}
		}
		assert gkao.length == V.size();
		
		// === Step 2: Main Perseus backup loop (until all beliefs improved) ===
		while (Btilde.size() > 0) {
			// Select a random belief point from those not yet improved
			int beliefIndex = rnd.nextInt(Btilde.size());
			BeliefPoint b = Btilde.get(beliefIndex);
			
			// Compute the backup (possibly ER) alpha vector at b
			AlphaVector alpha = backup(pomdp, immediateRewards, gkao, V, b);
			
			// Compute and compare value at b under old and new alpha vectors
			double oldValue = solver.AlphaVector.getValue(b.getBelief(), V);
			double newValue = alpha.getDotProduct(b.getBelief());
			
			if (newValue >= oldValue) {
				// New vector improves (or equals) value at b: accept into Vnext
				assert alpha.getAction() >= 0 && alpha.getAction() < pomdp.getNumActions() 
					: "invalid action: " + alpha.getAction();
				Vnext.add(alpha);
			} else {
				// No improvement: copy best existing alpha vector from V
				int bestVectorIndex = AlphaVector.getBestVectorIndex(b.getBelief(), V);
				assert V.get(bestVectorIndex).getAction() >= 0 && V.get(bestVectorIndex).getAction() < pomdp.getNumActions() 
					: "invalid action: " + V.get(bestVectorIndex).getAction();
				Vnext.add(V.get(bestVectorIndex));
			}
			
			// === Step 3: Update Btilde to only include beliefs not yet improved by Vnext ===
			List<BeliefPoint> newBtilde = new ArrayList<BeliefPoint>();
			for (BeliefPoint bp : B) {
				double oV = AlphaVector.getValue(bp.getBelief(), V);
				double nV = AlphaVector.getValue(bp.getBelief(), Vnext);
				if (nV < oV) {
					// Not yet improved: keep for another backup
					newBtilde.add(bp);
				}
			}
			Btilde = newBtilde;
		}
		
		// Return new value function (alpha vector set) after one backup stage
		return Vnext;
	}
	
	/**
	 * Computes the backup alpha vector at a given belief point.
	 * 
	 * This function implements the "backup" operation for point-based value iteration in the context of entropy-regularized POMDPs (ERPerseus).
	 * 
	 * Given:
	 *   - pomdp: the POMDP model
	 *   - immediateRewards: list of alpha vectors with immediate rewards for each action
	 *   - gkao: a 3D array of AlphaVectors, indexed by [k][action][observation], where k indexes value vectors in V; gkao[k][a][o] is the (a,o) one-step backup using V[k] as next-step value
	 *   - V: the current value function (set of alpha vectors); used to compute softmax weights at b' as V[k]·b'
	 *   - b: the belief point being considered
	 *
	 * The function proceeds as follows:
	 *   1. For each action:
	 *      a. For each observation:
	 *         - If using entropy regularization (lambda > 0): forms a softmax-weighted combination of gkao[k][a][o] over k, with weights softmax(V[k]·b' / λ).
	 *         - Otherwise: selects the best (maximum dot product) alpha vector from gkao for that action and observation.
	 *      b. Sums the observation-specific vectors for the action, scales by discount, and adds immediate rewards to form a candidate action alpha-vector.
	 *   2. After gathering candidate vectors (one per action), selects the final backup vector as follows:
	 *      - If using entropy regularization: forms a softmax-weighted combination of the action alpha-vectors, and tags the most probable action for compatibility.
	 *      - Otherwise: selects the action-alpha-vector that maximizes the value at the belief.
	 * 
	 * @return The backup alpha vector for use in the next value function
	 */
	private AlphaVector backup(POMDP pomdp, List<AlphaVector> immediateRewards, AlphaVector[][][] gkao, ArrayList<AlphaVector> V, BeliefPoint b) {
		int nStates = pomdp.getNumStates();
		int nActions = pomdp.getNumActions();
		int nObservations = pomdp.getNumObservations();
		
		// ga will hold one candidate alpha-vector per action (i.e., )
		List<AlphaVector> ga = new ArrayList<AlphaVector>(); // Will hold one candidate alpha-vector per action
		
		// For each possible action
		for(int action = 0; action < nActions; action++) {
			List<AlphaVector> oVectors = new ArrayList<AlphaVector>(); // Holds backed-up vectors, one per observation
			// For each possible observation
			for(int obs = 0; obs < nObservations; obs++) {

				// Because V (the collection of alpha vectors) is of size numActions, it is equivalent to ERPBVI's Q-function sets
				int K = gkao.length; // Number of Q-function sets, e.g., different prior value functions

				// For the current (action, observation), compute backup
				if (lambda > 0.0) {
					// --- Entropy-regularized backup: use softmax-weighted blend over all K (action,obs) continuations ---
					// Update belief: b' = Update(b, action, obs)
					BeliefPoint bPrime = pomdp.updateBelief(b, action, obs);
					double[] bPrimeBelief = (bPrime != null) ? bPrime.getBelief() : b.getBelief();
					
					// Compute softmax at updated belief b': weight by gkao[k][a][o]·b' (value of (a,o) continuation k at b')
					// gkao[k][a][o] is the backup for "take a, see o, get V[k]"; must use it, not V[k], for correct ER backup
					double[] dotProducts = new double[K];
					for(int k = 0; k < K; k++) {
						dotProducts[k] = gkao[k][action][obs].getDotProduct(bPrimeBelief) / lambda;
					}

					// The approximate entropy-regularised utility at the next belief-state b'
					/*double utility_bprime = 0.0;
					for (int k = 0; k < K; k++) {
						utility_bprime += Math.log(Math.exp(dotProducts[k])) - dotProducts[k];
					}
					utility_bprime /= lambda;
					*/

					// Since LogSumExp function is differentiable, we can compute the gradient of the utility, which
					// gives an alpha vector corresponding to the entropy regularised utility at the next belief
					double[] weights = softmax(dotProducts); // Softmax weights over Q-sets
					
					// Weighted sum of vectors using softmax weights (alpha_{a,o} in ERPBVI)
					double[] weightedSum = new double[nStates];
					for(int s = 0; s < nStates; s++) {
						weightedSum[s] = 0.0;
						for(int k = 0; k < K; k++) {
							weightedSum[s] += weights[k] * gkao[k][action][obs].getEntry(s);
						}
					}

					AlphaVector weightedVector = new AlphaVector(weightedSum);
					weightedVector.setAction(action);
					oVectors.add(weightedVector);
				} else {
					// --- Deterministic (classic) backup: pick hard max backup vector ---
					double maxVal = Double.NEGATIVE_INFINITY;
					AlphaVector maxVector = null;
					
					// Find the backup vector (over K value vectors) with the highest value at belief
					for(int k = 0; k < K; k++) {
						double product = gkao[k][action][obs].getDotProduct(b.getBelief());
						if(product > maxVal) {
							maxVal = product;
							maxVector = gkao[k][action][obs];
						}
					}
					
					assert maxVector != null;
					oVectors.add(maxVector);
				}
			}
			
			assert oVectors.size() > 0;

			// --- Sum over observation vectors to form backup for this action ---
			AlphaVector sumVector = oVectors.get(0);
			for(int j = 1; j < oVectors.size(); j++) {
				sumVector = AlphaVector.sumVectors(sumVector, oVectors.get(j));
			}
			
			// --- Multiply by discount factor (future rewards are discounted) ---
			double[] sumVectorEntries = sumVector.getEntries();
			for(int s = 0; s < nStates; s++) {
				sumVectorEntries[s] = pomdp.getDiscountFactor() * sumVectorEntries[s];
			}
			sumVector.setEntries(sumVectorEntries);
			
			// --- Add immediate rewards for this action to form overall backup vector ---
			AlphaVector av = AlphaVector.sumVectors(immediateRewards.get(action), sumVector);
			av.setAction(action);
			ga.add(av); // store action's candidate vector
		}
		
		assert ga.size() == nActions;
		
		// === Action selection at this belief: use softmax if entropy-regularized, else hard-max ===
		if (lambda > 0.0) {
			// --- Entropy-regularized: combine action vectors using softmax ---
			// 1. Compute Q-values (expected value at b, normalized by lambda) for all actions
			double[] qValues = new double[nActions];
			for(int action = 0; action < nActions; action++) {
				qValues[action] = ga.get(action).getDotProduct(b.getBelief()) / lambda;
			}
			
			// 2. Get softmax action probabilities
			double[] actionProbs = softmax(qValues);
			
			// 3. Weighted combination of all action vectors (distributional policy backup)
			double[] weightedAlpha = new double[nStates];
			for(int s = 0; s < nStates; s++) {
				weightedAlpha[s] = 0.0;
				for(int action = 0; action < nActions; action++) {
					weightedAlpha[s] += actionProbs[action] * ga.get(action).getEntry(s);
				}
			}
			AlphaVector vFinal = new AlphaVector(weightedAlpha);
			
			// 4. Set action label to most probable (for compatibility, e.g. in logging or policy extraction)
			int bestAction = 0;
			double bestProb = actionProbs[0];
			for(int action = 1; action < nActions; action++) {
				if(actionProbs[action] > bestProb) {
					bestProb = actionProbs[action];
					bestAction = action;
				}
			}
			vFinal.setAction(bestAction);
			return vFinal;
		} else {
			// --- Deterministic: choose the action with max expected value at b ---
			double maxVal = Double.NEGATIVE_INFINITY;
			AlphaVector vFinal = null;
			for(AlphaVector av : ga) {
				double product = av.getDotProduct(b.getBelief());
				if(product > maxVal) {
					maxVal = product;
					vFinal = av;
				}
			}
			assert vFinal != null;
			return vFinal;
		}
	}

	/**
	 * Main solver method: Implements the ERPerseus (Entropy-Regularized Perseus) algorithm
	 * to solve a POMDP using point-based value iteration.
	 * 
	 * The algorithm iteratively improves a value function V (represented as a set of alpha vectors)
	 * by performing backups at sampled belief points until convergence.
	 * 
	 * Algorithm overview:
	 *   1. Sample reachable belief points B from the POMDP
	 *   2. Initialize value function V with immediate rewards (one vector per action)
	 *   3. Iteratively apply backup stages until convergence:
	 *      - backupStage() improves V at all beliefs in B
	 *      - Check if value improvement is below tolerance
	 *   4. Return the converged value function V
	 * 
	 * @param pomdp The POMDP to solve
	 * @return ArrayList of AlphaVectors representing the converged value function
	 */
	public ArrayList<AlphaVector> solve(POMDP pomdp) {		
		int nStates = pomdp.getNumStates();
		int nActions = pomdp.getNumActions();
		
		System.out.println();
		System.out.println("=== RUN POMDP SOLVER ===");
		System.out.println("Algorithm: ERPerseus (point-based value iteration)");
		System.out.println("Belief sampling started...");
		
		// === Step 1: Sample reachable belief points ===
		// Generate a set B of belief points by simulating random trajectories through the POMDP.
		// These beliefs represent states the agent might actually reach, focusing computation
		// on relevant parts of the belief space rather than the entire (infinite) space.
		ArrayList<BeliefPoint> B = getBeliefPoints(pomdp);
		System.out.println("Number of beliefs: "+B.size()+"\n");
		
		// === Step 2: Initialize value function V ===
		// Start with immediate rewards: for each action, create an alpha vector where each entry
		// is the reward for that state-action pair. This gives a lower bound on the value function.
		// Both V and immediateRewards are initialized identically, but immediateRewards stays
		// constant (used in backups) while V gets updated during value iteration.
		ArrayList<AlphaVector> V = new ArrayList<AlphaVector>();
		ArrayList<AlphaVector> immediateRewards = new ArrayList<AlphaVector>();
		
		for(int a = 0; a < nActions; a++) {
			// Create alpha vector with immediate rewards for this action across all states
			double[] entries = new double[nStates];
			for(int s = 0; s < nStates; s++) {
				entries[s] = pomdp.getReward(s, a);
			}
			AlphaVector av = new AlphaVector(entries);
			av.setAction(a);
			V.add(av);  // Add to value function (will be updated during iteration)
			immediateRewards.add(av);  // Keep constant copy for use in backups
		}
				
		// === Step 3: Iterative value improvement (main value iteration loop) ===
		// Perform backup stages until the value function converges or time limit is reached.
		// Each backup stage improves the value function V at all sampled belief points in B.
		long startTime = System.currentTimeMillis();
		while(true) {
			// Apply one backup stage: improves V at all beliefs in B using Perseus algorithm
			// Returns Vnext, the new (hopefully improved) value function
			ArrayList<AlphaVector> Vnext = backupStage(pomdp, immediateRewards, V, B);
			
			// Measure convergence: compute maximum value improvement across all beliefs
			// If improvement is below tolerance, the value function has converged
			double valueDifference = getValueDifference(B, V, Vnext);
			
			// Update value function for next iteration
			V = Vnext;
			
			// Check convergence criteria: stop if value improvement is small enough,
			// or if we've exceeded the time limit
			double elapsedTime = (System.currentTimeMillis() - startTime) * 0.001;
			if(valueDifference < sp.getValueFunctionTolerance() || elapsedTime > sp.getTimeLimit()) {
				break;
			}
		}
		
		// === Step 4: Finalize and return results ===
		totalSolveTime = (System.currentTimeMillis() - startTime);
		
		// Compute expected value at initial belief using the converged value function
		expectedValue = AlphaVector.getValue(pomdp.getInitialBelief().getBelief(), V);
		
		// Find best action at initial belief (for reporting purposes)
		int bestIndex = AlphaVector.getBestVectorIndex(pomdp.getInitialBelief().getBelief(), V);
		
		// Print results
		System.out.print("Initial Belief: ");
		for(int i = 0; i < pomdp.getInitialBelief().getBelief().length; i++) {
			System.out.print(pomdp.getInitialBelief().getBelief()[i]+" ");
		}
		System.out.println("\nSelected Action: " + V.get(bestIndex).getAction());
		
		// Save value function to file for analysis/debugging
		File outputDir = new File(sp.getOutputDir());
		File outputFile = new File(outputDir, pomdp.getInstanceName() + ".alpha");
		OutputFileWriter.dumpValueFunction(pomdp, V, outputFile.getAbsolutePath(), sp.dumpActionLabels(), true);
		
		System.out.println("Value function being returned, which is of type ArrayList<AlphaVector>: " + V);
		return V;
	}

	/**
	 * Computes the maximum value improvement between two consecutive value functions
	 * across all sampled belief points.
	 * 
	 * This is used as a convergence measure: when the maximum improvement falls below
	 * a tolerance threshold, the value iteration has effectively converged.
	 * 
	 * @param B      List of sampled belief points to evaluate at
	 * @param V      Previous value function (set of alpha vectors)
	 * @param Vnext  New value function (set of alpha vectors) after backup stage
	 * @return Maximum value improvement: max_b [Vnext(b) - V(b)] across all beliefs in B
	 */
	private double getValueDifference(List<BeliefPoint> B, ArrayList<AlphaVector> V, ArrayList<AlphaVector> Vnext) {
		double maxDifference = Double.NEGATIVE_INFINITY;
		
		// Check value improvement at each sampled belief point
		for(BeliefPoint b : B) {
			// Compute value difference: new value minus old value at this belief
			double diff = AlphaVector.getValue(b.getBelief(), Vnext) - AlphaVector.getValue(b.getBelief(), V);
			if(diff > maxDifference) maxDifference = diff;
		}
		
		return maxDifference;
	}
	
	/**
	 * Compute softmax weights: softmax(x_i) = exp(x_i) / sum(exp(x_j))
	 * Uses numerically stable version with max subtraction
	 * 
	 * @param x the input array of values
	 * @return array of softmax probabilities
	 */
	private double[] softmax(double[] x) {
		double[] result = new double[x.length];
		
		// Find max for numerical stability
		double max = Double.NEGATIVE_INFINITY;
		for (double v : x) {
			if (v > max) max = v;
		}
		
		// Compute exp and sum
		double sum = 0.0;
		for (int i = 0; i < x.length; i++) {
			result[i] = Math.exp(x[i] - max);
			sum += result[i];
		}
		
		// Normalize
		for (int i = 0; i < x.length; i++) {
			result[i] /= sum;
		}
		
		return result;
	}
	
	/**
	 * Get expected value of the solution
	 * @return expected value
	 */
	public double getExpectedValue() {
		return expectedValue;
	}
}
