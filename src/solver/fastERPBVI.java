/*******************************************************************************
 * Entropy-Regularized Point-Based Value Iteration (ERPBVI)
 * Based on "Entropy-regularized Point-based Value Iteration" paper
 *******************************************************************************/

package solver;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.Set;

import pomdp.POMDP;
import pomdp.SolverProperties;

/**
 * ERPBVI Solver - Entropy-Regularized Point-Based Value Iteration
 * 
 * Uses soft-max (log-sum-exp) value functions instead of hard max
 * for smoother policy learning.
 */
public class fastERPBVI implements Solver {    
    private Random rnd;
    private SolverProperties sp;
    private double lambda;  // temperature parameter
    private boolean verbose;

    private long totalSolveTime = 0;
    private double expectedValue;
    
    // Q-functions: Ξ[action] = list of alpha vectors for that action
    private List<List<AlphaVector>> Xi;
    
    // Profiling counters (in nanoseconds)
    private long totalBeliefUpdateTime = 0;
    private long totalArgmaxTime = 0;
    private long totalSoftmaxTime = 0;
    private long totalLookaheadTime = 0;
    private int backupCount = 0;
    
    // Cache for belief updates: key = (belief, action, observation)
    private Map<String, BeliefPoint> beliefUpdateCache = new HashMap<>();

    public fastERPBVI(SolverProperties solverProperties, Random rnd, double lambda, boolean verbose) {
        this.sp = solverProperties;
        this.rnd = rnd;
        this.lambda = lambda;
        this.verbose = verbose;
    }
    
    @Override
    public String getType() {
        return "approximate-erpbvi";
    }
    
    @Override
    public double getTotalSolveTime() {
        return totalSolveTime * 0.001;
    }
    
    @Override
    public double getExpectedValue() {
        return expectedValue;
    }
    
    // ==================== Utility Methods ====================
    
    /**
     * Compute softmax weights: softmax(x_i) = exp(x_i) / sum(exp(x_j))
     * Uses numerically stable version with max subtraction
     */
    /**
     * Compute softmax weights: softmax(x_i) = exp(x_i) / sum(exp(x_j))
     * Uses numerically stable version with max subtraction
     * 
     * @param x the input array of values (typically action-obs-values or scores), over which to compute softmax
     * @return array of softmax probabilities corresponding to each input
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
     * Find the alpha vector with maximum dot product at belief b.
     * 
     * @param Gamma List of AlphaVectors (value function vectors) to search through.
     * @param b     Belief point (probability distribution over states).
     * @return      The AlphaVector in Gamma that achieves the maximum value at b.
     */
    private AlphaVector argmaxAlpha(List<AlphaVector> Gamma, double[] b) {
        AlphaVector best = null;
        double bestValue = Double.NEGATIVE_INFINITY;
        
        for (AlphaVector alpha : Gamma) {
            double value = alpha.getDotProduct(b);
            if (value > bestValue) {
                bestValue = value;
                best = alpha;
            }
        }
        
        return best;
    }
    
    /**
     * Compute value at belief using standard max
     */
    private double computeValue(double[] b, List<AlphaVector> Gamma) {
        double maxVal = Double.NEGATIVE_INFINITY;
        for (AlphaVector alpha : Gamma) {
            double val = alpha.getDotProduct(b);
            if (val > maxVal) maxVal = val;
        }
        return maxVal;
    }
    
    /**
     * Generate cache key for belief update
     */
    private String getBeliefUpdateKey(BeliefPoint b, int a, int o) {
        return beliefKey(b.getBelief()) + "|a" + a + "|o" + o;
    }
    
    // ==================== Algorithm 3: Backup ====================
    
    /**
     * Backup a single belief point (Algorithm 3)
     * Returns new alpha vectors for each action
     * @param newAlphas is the set of all Q-functions. And the Q-function for action a is denoted by the collection of alpha-vectors Gamma_a. So each Q-function has a collection of alpha vectors,and this is a collection of them
     */
    /**
     * Backup a single belief point (Algorithm 3)
     * Appends new alpha vector to each Q-function Ξ[a]
     * 
     * In the ERPBVI backup, lambda controls policy stochasticity:
     *   - Large lambda: more stochastic, encourages exploration; weights in softmax are more uniform.
     *   - Small lambda: more deterministic/greedy, selects the action(s) with highest expected value.
     * 
     * Since the POMDP model is only an approximation of reality and our beliefs may be incorrect or evolve,
     * using *some* stochasticity (larger lambda) can be valuable. It encourages exploration and helps prevent
     * the algorithm from committing too early to suboptimal actions based on limited or imperfect knowledge.
     * This, in turn, enables better updating of our POMDP beliefs as more observations are made.
     * 
     * If you always choose the best action (lambda→0), you risk premature convergence and less robust belief updates.
     * A moderate lambda (not too small, not too large) is typically best, balancing exploitation and exploration.
     * 
     * You can start with a value such as lambda=1.0 or higher, and tune it based on empirical performance.
     */
    private void backup(POMDP pomdp, BeliefPoint b) {
        int nStates = pomdp.getNumStates();
        int nActions = pomdp.getNumActions();
        int nObservations = pomdp.getNumObservations();
        double gamma = pomdp.getDiscountFactor();

        long beliefUpdateTime = 0;
        long argmaxTime = 0;
        long softmaxTime = 0;
        long lookaheadTime = 0;
        long startTime;

        // For each action a, compute α_a and append to Ξ[a]
        for (int a = 0; a < nActions; a++) {
            // For each observation, compute the soft-weighted alpha
            double[][] alphaAO = new double[nObservations][nStates];
            
            for (int o = 0; o < nObservations; o++) {
                // Update belief: b' = Update(b, a, o) - use cache if available
                startTime = System.nanoTime();
                String cacheKey = getBeliefUpdateKey(b, a, o);
                BeliefPoint bPrimePoint = beliefUpdateCache.get(cacheKey);
                if (bPrimePoint == null) {
                    bPrimePoint = pomdp.updateBelief(b, a, o);
                    if (bPrimePoint != null) {
                        beliefUpdateCache.put(cacheKey, bPrimePoint);
                    }
                }
                double[] bPrime = (bPrimePoint != null) ? bPrimePoint.getBelief() : null;
                beliefUpdateTime += System.nanoTime() - startTime;
                
                if (bPrime == null) {
                    // Observation not possible, use zeros
                    for (int s = 0; s < nStates; s++) {
                        alphaAO[o][s] = 0.0;
                    }
                    continue;
                }
                
                // For each Q-function Γi in Ξ, find argmax alpha at b' (eq 6)
                // A_ao contains the best alpha from each Q-function
                startTime = System.nanoTime();
                List<AlphaVector> A_ao = new ArrayList<>();
                for (List<AlphaVector> Gamma_i : Xi) {
                    AlphaVector bestAlpha = argmaxAlpha(Gamma_i, bPrime);
                    if (bestAlpha != null) {
                        A_ao.add(bestAlpha);
                    }
                }
                argmaxTime += System.nanoTime() - startTime;
                
                if (A_ao.isEmpty()) {
                    for (int s = 0; s < nStates; s++) {
                        alphaAO[o][s] = 0.0;
                    }
                    continue;
                }
                
                // Compute softmax weights: SoftMax(A_ao^T * b' / λ)
                startTime = System.nanoTime();
                double[] dotProducts = new double[A_ao.size()];
                for (int i = 0; i < A_ao.size(); i++) {
                    dotProducts[i] = A_ao.get(i).getDotProduct(bPrime) / lambda;
                }
                double[] weights = softmax(dotProducts);
                
                // Compute soft-weighted alpha: α_ao = A_ao * weights (eq 8)
                for (int s = 0; s < nStates; s++) {
                    double sum = 0.0;
                    for (int i = 0; i < A_ao.size(); i++) {
                        sum += weights[i] * A_ao.get(i).getEntry(s);
                    }
                    alphaAO[o][s] = sum;
                }
                softmaxTime += System.nanoTime() - startTime;
            }
            
            // Compute lookahead: α_a = R(s,a) + γ * Σ T(s'|s,a) * O(o|s',a) * α_ao (eq 9)
            startTime = System.nanoTime();
            double[] alphaA = new double[nStates];
            for (int s = 0; s < nStates; s++) {
                double value = pomdp.getReward(s, a);
                
                double futureValue = 0.0;
                for (int sPrime = 0; sPrime < nStates; sPrime++) {
                    double transProb = pomdp.getTransitionProbability(s, a, sPrime);
                    
                    for (int o = 0; o < nObservations; o++) {
                        double obsProb = pomdp.getObservationProbability(a, sPrime, o);
                        futureValue += transProb * obsProb * alphaAO[o][sPrime];
                    }
                }
                
                alphaA[s] = value + gamma * futureValue;
            }
            lookaheadTime += System.nanoTime() - startTime;
            
            // Append α_a to Q-function Ξ[a] (line 9 of Algorithm 3)
            AlphaVector newAlpha = new AlphaVector(alphaA);
            newAlpha.setAction(a);
            Xi.get(a).add(newAlpha);
        }
        
        // Accumulate timing stats
        totalBeliefUpdateTime += beliefUpdateTime;
        totalArgmaxTime += argmaxTime;
        totalSoftmaxTime += softmaxTime;
        totalLookaheadTime += lookaheadTime;
        backupCount++;
    }
    
    // ==================== Algorithm 2: Improve ====================
    
    /**
     * Improve Q-functions until convergence (Algorithm 2)
     * Modifies Xi directly by appending new alpha vectors.
     * Now has a safeguard: maxInnerIterations to avoid infinite loops.
     */
    private void improve(POMDP pomdp, List<BeliefPoint> B) {
        long startTime = System.currentTimeMillis();
        
        // Clear belief update cache at start of improve iteration
        beliefUpdateCache.clear();
        
        while (true) {
            // Store old values for convergence check
            List<List<AlphaVector>> XiOld = new ArrayList<>();
            for (List<AlphaVector> Gamma : Xi) {
                XiOld.add(new ArrayList<>(Gamma));
            }

            // Backup each belief point (Algorithm 2, line 3)
            for (BeliefPoint b : B) {
                backup(pomdp, b);  // Appends to Xi directly
            }

            // Check convergence
            double maxGap = 0.0;
            for (BeliefPoint b : B) {
                for (int a = 0; a < pomdp.getNumActions(); a++) {
                    double oldValue = computeValue(b.getBelief(), XiOld.get(a));
                    double newValue = computeValue(b.getBelief(), Xi.get(a));
                    double gap = Math.abs(newValue - oldValue);
                    if (gap > maxGap) maxGap = gap;
                }
            }

            double elapsedTime = (System.currentTimeMillis() - startTime) * 0.001;
            if (verbose) {
                System.out.println("    Improving alphas, maximum gap: " + maxGap + " (time elapsed " + elapsedTime + " seconds)");
            }

            if (maxGap <= sp.getValueFunctionTolerance() || elapsedTime > sp.getTimeLimit()) {
                break;
            }
        }
    }
    
    // ==================== Pruning ====================
    
    /**
     * Simple pruning: remove pointwise dominated vectors
     */
    private void prune() {
        for (int a = 0; a < Xi.size(); a++) {
            List<AlphaVector> Gamma = Xi.get(a);
            ArrayList<AlphaVector> pruned = new ArrayList<>();
            
            for (AlphaVector alpha : Gamma) {
                if (!alpha.isPointwiseDominated(pruned)) {
                    pruned.add(alpha);
                }
            }
            
            Xi.set(a, pruned);
        }
    }
    
    // ==================== Expand ====================
    
    /**
     * Expand belief set with successor beliefs
     * Optimized with early termination and distance threshold
     */
    private List<BeliefPoint> expand(POMDP pomdp, List<BeliefPoint> B, Set<BeliefPoint> Bset) {
        List<BeliefPoint> BNew = new ArrayList<>(B);
        double distanceThreshold = 0.01; // Minimum distance to consider adding a belief
        
        int addedCount = 0;
        int maxAdditionsPerIteration = Math.max(1, B.size() / 2); // Limit expansion rate
        
        for (BeliefPoint b : B) {
            if (addedCount >= maxAdditionsPerIteration) {
                break; // Early termination: limit expansion
            }
            
            // Find most distant successor
            double maxDist = -1;
            BeliefPoint bestSuccPoint = null;
            
            for (int a = 0; a < pomdp.getNumActions(); a++) {
                for (int o = 0; o < pomdp.getNumObservations(); o++) {
                    BeliefPoint bPrimePoint = pomdp.updateBelief(b, a, o);
                    
                    if (bPrimePoint != null && !Bset.contains(bPrimePoint)) {
                        // Compute minimum distance to existing beliefs
                        double dist = computeMinDistance(bPrimePoint.getBelief(), B);
                        if (dist > maxDist && dist > distanceThreshold) {
                            maxDist = dist;
                            bestSuccPoint = bPrimePoint;
                        }
                    }
                }
            }
            
            if (bestSuccPoint != null) {
                BeliefPoint newBelief = new BeliefPoint(bestSuccPoint.getBelief());
                BNew.add(newBelief);
                Bset.add(newBelief);
                addedCount++;
            }
        }
        
        return BNew;
    }
    
    private String beliefKey(double[] b) {
        StringBuilder sb = new StringBuilder();
        for (double v : b) {
            sb.append(String.format("%.6f,", v));
        }
        return sb.toString();
    }
    
    private double computeMinDistance(double[] b, List<BeliefPoint> B) {
        double minDist = Double.POSITIVE_INFINITY;
        for (BeliefPoint bp : B) {
            double dist = 0;
            for (int i = 0; i < b.length; i++) {
                dist += Math.abs(b[i] - bp.getBelief(i));
            }
            if (dist < minDist) minDist = dist;
        }
        return minDist == Double.POSITIVE_INFINITY ? 0 : minDist;
    }
    
    // ==================== Algorithm 1: Solve ====================
    /**
     * Solves the given POMDP using the Entropy-Regularized Point-Based Value Iteration (ERPBVI) algorithm.
     *
     * This function implements the main loop of the ERPBVI algorithm. It initializes the belief set with the
     * initial belief and corner (deterministic state) beliefs, sets up the initial Q-value lower bound, and
     * iteratively improves the value function using point-based backups. The value function is represented by
     * sets of alpha-vectors for each action, and is regularized using an entropy term (controlled by lambda).
     * The main steps are:
     *   - Initialize beliefs and value function bounds
     *   - For a given number of iterations or until convergence:
     *       - Perform a point-based backup at each belief to update Q-values and alpha-vectors
     *       - Expand the belief set with reachable beliefs if desired
     *       - Compute error/bellman residual to check convergence
     *
     * The result is a list of alpha-vectors representing the approximate value function, which can be used to
     * define a stochastic policy via softmax over the Q-values.
     *
     * @param pomdp The POMDP problem instance to solve
     * @return An ArrayList of AlphaVectors representing the optimized value function
     */
    @Override
    public ArrayList<AlphaVector> solve(POMDP pomdp) {
        int nStates = pomdp.getNumStates();
        int nActions = pomdp.getNumActions();
        double gamma = pomdp.getDiscountFactor();
        
        // Reset profiling counters
        totalBeliefUpdateTime = 0;
        totalArgmaxTime = 0;
        totalSoftmaxTime = 0;
        totalLookaheadTime = 0;
        backupCount = 0;
        
        System.out.println();
        System.out.println("=== RUN ERPBVI SOLVER ===");
        System.out.println("Algorithm: Entropy-Regularized Point-Based Value Iteration");
        System.out.println("Parameters: beliefSamplingRuns=" + sp.getBeliefSamplingRuns() + 
                          ", beliefSamplingSteps=" + sp.getBeliefSamplingSteps() + 
                          ", epsilon=" + sp.getValueFunctionTolerance() + 
                          ", lambda=" + lambda +
                          ", verbose=" + verbose);
        
        // Initialize beliefs: B ← {b0}
        List<BeliefPoint> B = new ArrayList<BeliefPoint>();
        Set<BeliefPoint> Bset = new HashSet<BeliefPoint>();
        B.add(pomdp.getInitialBelief());
        Bset.add(pomdp.getInitialBelief());
        
        // Add corner beliefs
        for (int s = 0; s < nStates; s++) {
            double[] corner = new double[nStates];
            corner[s] = 1.0;
            BeliefPoint cornerBelief = new BeliefPoint(corner);
            if (!Bset.contains(cornerBelief)) {
                B.add(cornerBelief);
                Bset.add(cornerBelief);
            }
        }
        
        if (verbose) {
            System.out.println("Initial beliefs: " + B.size());
        }
        
        // Initialize Q-values: Ξ ← InitializeQValues(P)
        // Lower bound: 1/(1-γ) * max_a min_s R(s,a)
        double alphaInit = Double.NEGATIVE_INFINITY;
        for (int a = 0; a < nActions; a++) {
            double minReward = Double.POSITIVE_INFINITY;
            for (int s = 0; s < nStates; s++) {
                double r = pomdp.getReward(s, a);
                if (r < minReward) minReward = r;
            }
            if (minReward > alphaInit) alphaInit = minReward;
        }
        alphaInit = alphaInit / (1.0 - gamma);
        
        Xi = new ArrayList<>();
        for (int a = 0; a < nActions; a++) {
            List<AlphaVector> Gamma = new ArrayList<>();
            double[] entries = new double[nStates];
            for (int s = 0; s < nStates; s++) {
                entries[s] = alphaInit;
            }
            AlphaVector initAlpha = new AlphaVector(entries);
            initAlpha.setAction(a);
            Gamma.add(initAlpha);
            Xi.add(Gamma);
        }
        
        // Main loop
        long startTime = System.currentTimeMillis();
        int prevSize = B.size();
        
        for (int i = 1; i <= sp.getBeliefSamplingRuns(); i++) {
            // Improve
            improve(pomdp, B);
            
            // Prune
            prune();
            
            // Expand
            B = expand(pomdp, B, Bset);
            
            if (verbose) {
                System.out.println("Iteration " + i + ": " + B.size() + " beliefs");
            }
            
            // Early termination if belief space didn't expand
            if (B.size() == prevSize) {
                if (verbose) {
                    System.out.println("Belief space did not expand. Terminating early.");
                }
                break;
            }
            prevSize = B.size();
        }
        
        totalSolveTime = System.currentTimeMillis() - startTime;
        
        // Collect all alpha vectors for return
        ArrayList<AlphaVector> allAlphas = new ArrayList<>();
        for (List<AlphaVector> Gamma : Xi) {
            allAlphas.addAll(Gamma);
        }
        
        expectedValue = AlphaVector.getValue(pomdp.getInitialBelief().getBelief(), allAlphas);
        
        System.out.println("Solve time: " + getTotalSolveTime() + " seconds");
        System.out.println("Total alpha vectors: " + allAlphas.size());
        System.out.println("Expected value: " + expectedValue);
        
        // Print profiling results
        System.out.println();
        System.out.println("=== ERPBVI PROFILING ===");
        System.out.println("Total backups: " + backupCount);
        double totalBackupTime = (totalBeliefUpdateTime + totalArgmaxTime + totalSoftmaxTime + totalLookaheadTime) / 1e9;
        System.out.println("Total backup computation time: " + String.format("%.3f", totalBackupTime) + " seconds");
        System.out.println("  Belief update:  " + String.format("%.3f", totalBeliefUpdateTime / 1e9) + "s (" + 
            String.format("%.1f", 100.0 * totalBeliefUpdateTime / (totalBeliefUpdateTime + totalArgmaxTime + totalSoftmaxTime + totalLookaheadTime)) + "%)");
        System.out.println("  Argmax search:  " + String.format("%.3f", totalArgmaxTime / 1e9) + "s (" + 
            String.format("%.1f", 100.0 * totalArgmaxTime / (totalBeliefUpdateTime + totalArgmaxTime + totalSoftmaxTime + totalLookaheadTime)) + "%)");
        System.out.println("  Softmax (eq8):  " + String.format("%.3f", totalSoftmaxTime / 1e9) + "s (" + 
            String.format("%.1f", 100.0 * totalSoftmaxTime / (totalBeliefUpdateTime + totalArgmaxTime + totalSoftmaxTime + totalLookaheadTime)) + "%)");
        System.out.println("  Lookahead(eq9): " + String.format("%.3f", totalLookaheadTime / 1e9) + "s (" + 
            String.format("%.1f", 100.0 * totalLookaheadTime / (totalBeliefUpdateTime + totalArgmaxTime + totalSoftmaxTime + totalLookaheadTime)) + "%)");
        System.out.println();
        
        return allAlphas;
    }
    
    /**
     * Get the Q-functions (for use with EntropyRegularizedPolicy)
     */
    public List<List<AlphaVector>> getQFunctions() {
        return Xi;
    }
    
    /**
     * Get lambda (temperature parameter)
     */
    public double getLambda() {
        return lambda;
    }
}
