package tud.ke.ml.project.classifier;

import tud.ke.ml.project.util.Pair;

import java.io.Serializable;
import java.util.HashMap;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;

/**
 * This implementation assumes the class attribute is always available (but probably not set).
 *
 */
public class NearestNeighbor extends INearestNeighbor implements Serializable {
	private static final long serialVersionUID = 1L;

	protected double[] scaling;
	protected double[] translation;

	LinkedList<LinkedList<Object>> instances;

	// TODO remove debug stuff from final version?
	public static boolean debug = false;

	@Override
	public String getMatrikelNumbers() {
		return "2484022,2605351,2650348";
	}

	@Override
	protected void learnModel(List<List<Object>> data) {
		if(debug)
			System.out.println("Learning Model with Data size: " + data.size());

		instances = new LinkedList<LinkedList<Object>>();

		// copy the instances from data into instances
		for(List<Object> dataInstance : data){
			LinkedList<Object> instance = new LinkedList<Object>();

			// copy the single attribute 
			// we create clones to be independent from the original object
			for(Object attribute : dataInstance){
				if(attribute instanceof String)
					instance.add(new String((String)attribute));
				else if(attribute instanceof Double)
					instance.add(new Double((double)attribute));
				else
					throw new IllegalArgumentException("Attribute does not have a known Type!");
			}

			// add new instance to local instance list
			instances.add(instance);
		}

		if(debug){
			System.out.println("Saved " + instances.size());
			for(LinkedList<Object> instance : instances){
				System.out.println(instance);
			}
		}
	}

	@Override
	protected Map<Object, Double> getUnweightedVotes(List<Pair<List<Object>, Double>> subset) {
		HashMap<Object, Double> result = new HashMap<Object, Double>();

		for(Pair<List<Object>, Double> pair : subset){
			// get the class attribute of the given instance
			Object classAttribute = pair.getA().get(getClassAttribute());

			// add one to the votes count
			if(result.containsKey(classAttribute))
				result.put(classAttribute, result.get(classAttribute) + 1);
				// this is the first occurrence of the given class
			else
				result.put(classAttribute, 1.0);
		}
		return result;
	}

	/**
	 * Collects the votes based on the inverse distance weighting schema
	 *
	 * @param subset Set of nearest neighbors with their distance
	 * @return Map of classes with their votes (e.g. returnValue.get("yes") are the votes for class "yes")
	 */
	@Override
	protected Map<Object, Double> getWeightedVotes(List<Pair<List<Object>, Double>> subset) {

		// HashMap with the sum of all inverse distances
		HashMap<Object, Double> result = new HashMap<Object, Double>();

		for(Pair<List<Object>, Double> pair : subset){
			// get the class attribute of the given instance
			Object classAttribute = pair.getA().get(getClassAttribute());

			// Get the distance of the given instance
			double distance = pair.getB().doubleValue();

			// bigger distance --> smaller weight (See Instanzlernen slide 9)
			double weightedDistance = 1 / (distance + .0001);

			// add one to the votes count
			if(result.containsKey(classAttribute))
				result.put(classAttribute, result.get(classAttribute) + weightedDistance);
				// this is the first occurrence of the given class
			else
				result.put(classAttribute, weightedDistance);
		}
		return result;
	}

	@Override
	protected Object getWinner(Map<Object, Double> votes) {
		// TODO exception handling?

		// return the class with the most votes
		LinkedList<Object> maxValues = new LinkedList<Object>();
		Double votesMax = null;

		// get the object with the most votes
		for(Object o : votes.keySet()){
			Double d = votes.get(o);

			if(votesMax == null || d > votesMax){
				votesMax = d;
				maxValues.clear();
				maxValues.add(o);
			}else if(d == votesMax){
				maxValues.add(o);
			}
		}
		if(maxValues.size()>1)
			System.out.println("size: " + maxValues.size());

		// TODO select best value
		return selectWinner(maxValues);
	}

	private Object selectWinner(LinkedList<Object> winnerCandidates){

		if(winnerCandidates.size()>1){
			LinkedList<LinkedList<Object>> winnerInstances = new LinkedList<LinkedList<Object>>();

			for(final Object winner : winnerCandidates){
				LinkedList<Object> correspondingInstance = instances.stream().filter(instance -> instance.contains(winner)).findFirst().get();
				winnerInstances.add(correspondingInstance);
			}



			return winnerCandidates.getFirst();
		}

		return winnerCandidates.getFirst();
	}

	@Override
	protected Object vote(List<Pair<List<Object>, Double>> subset) {
		Map<Object, Double> voteResult;
		if(isInverseWeighting()) {
			// get the voting results
			voteResult = getWeightedVotes(subset);
		} else {
			// get the voting results
			voteResult = getUnweightedVotes(subset);
		}
		// and return the winner
		return getWinner(voteResult);
	}

	@Override
	protected List<Pair<List<Object>, Double>> getNearest(List<Object> data) {
		LinkedList<Pair<List<Object>, Double>> result = new LinkedList<Pair<List<Object>, Double>>();

		// debug prefix
		String nnPrefix = "NN > ";
		String nnInsertPrefix = "NN > \tInsert > ";

		if(debug)
			System.out.println(nnPrefix + "Calculating NN (k="+getkNearest()+")to " + data);

		// get the scaling factors; if normalization is off,
		// the scaling and translation will contain the identities
		// therefore without any effect while normalizing.
		double[][] factors = normalizationScaling();
		scaling = factors[0].clone();
		translation = factors[1].clone();


		// get the nearest instances
		for(List<Object> instance : instances){
			Double distance;
			if(getMetric() == 0) {
				distance = determineManhattanDistance(instance, data);
			} else {
				distance = determineEuclideanDistance(instance, data);
			}
			Pair<List<Object>, Double> instancePair = new Pair<List<Object>, Double>(instance, distance);

			if(debug)
				System.out.println(nnPrefix + "Distance of " + instance + " is " + distance);

			// add the element when the result list is empty
			// and add it as the last one when the result list contains less than k 
			// instances and the distance is greater than the last element in the list
			if(result.size() == 0 || distance >= result.get(result.size() - 1).getB()){
				if(debug){
					if(result.size() > 0){
						System.out.println(nnInsertPrefix + "Distance of last instance: " + result.get(result.size() - 1).getB());
					}
				}
				// only add a too distant element when the list is not full yet
				if(result.size() < getkNearest()){
					if(debug)
						System.out.println(nnInsertPrefix + "Added directly");
					result.addLast(instancePair);
				}
				// otherwise (distance is larger than the last element in the list
				// and number of elements is >= k): Ignore the instance
				else{
					if(debug)
						System.out.println(nnInsertPrefix + "Did not add because the distance is too big");
				}
			}
			else {
				// the distance is smaller than the last element in the result list
				// -> we need to sort it inside of the list

				// start from bottom up
				// determine the position where we want to insert the new instance
				// TODO: Use LinkedList methods for increased performance
				int i;
				for(i = 0; i < result.size(); i++){
					if(debug)
						System.out.println(nnInsertPrefix + i + ": "+result.get(i).getB() + " > " + distance + " ?");

					if(result.get(i).getB() > distance){
						// we found an index which instances distance is bigger than ours
						break;
					}
				}
				if(debug)
					System.out.println(nnInsertPrefix + "Index to insert: " + i);

				result.add(i, instancePair);

				if(debug){
					System.out.println(nnPrefix + "-- Temporary Result for k=" + getkNearest());
					for(int j = 0; j < result.size(); j++){
						Pair<List<Object>, Double> pair = result.get(j);
						System.out.println(nnPrefix + j + ": " + pair.getA() + "\t" + pair.getB());
					}
				}

				// when we exceeded the list size: remove the instance with the 
				// largest distance (the last one in the list)
				if(result.size() > getkNearest()){
					result.removeLast();
					if(debug)
						System.out.println(nnPrefix + "Removed last instance.");
				}
			}
		}

		if(debug){
			System.out.println(nnPrefix + "Result of NN with k=" + getkNearest());
			System.out.println(nnPrefix + "Raking | Instance | Distance");
			for(int j = 0; j < result.size(); j++){
				Pair<List<Object>, Double> pair = result.get(j);
				System.out.println(nnPrefix + j + ": " + pair.getA() + "\t" + pair.getB());
			}
		}

		return result;
	}

	@Override
	protected double determineManhattanDistance(List<Object> instance1, List<Object> instance2) {
		// TODO exception handling?

		double distance = 0;
		int size = instance1.size();
		for(int i = 0; i < size; i++){
			// ignore the class attribute in the distance calculation!
			if(i != getClassAttribute() && !instance1.get(i).equals(instance2.get(i))){
				if(instance1.get(i) instanceof Double && instance2.get(i) instanceof Double){
					double v1 = ((Double)instance1.get(i) + translation[i])*scaling[i];
					double v2 = ((Double)instance2.get(i) + translation[i])*scaling[i];

					distance += Math.abs(v1 - v2);
				}else{
					distance += 1;
				}
			}
		}

		return distance;
	}

	@Override
	protected double determineEuclideanDistance(List<Object> instance1, List<Object> instance2) {
		double distance = 0;
		int size = instance1.size();

		for(int i = 0; i < size; i++){

			// ignore the class attribute in the distance calculation!
			if(i != getClassAttribute() && !instance1.get(i).equals(instance2.get(i))){
				if(instance1.get(i) instanceof Double && instance2.get(i) instanceof Double){
					double v1 = ((Double)instance1.get(i) + translation[i])*scaling[i];
					double v2 = ((Double)instance2.get(i) + translation[i])*scaling[i];

					distance += Math.abs(v1 - v2) * Math.abs(v1 - v2);
				} else{
					distance += 1;
				}
			}
		}

		return Math.sqrt(distance);
	}

	@Override
	protected double[][] normalizationScaling() {

		int numA = this.instances.getFirst().size();
		// save minimum and maximum per attribute
		double[][] boundsPerAttr = new double[2][numA];
		for (List<Object> instance : this.instances) {
			int i = 0;
			for (Object attr : instance) {
				if (attr instanceof Double) {
					boundsPerAttr[0][i] = (double) attr < boundsPerAttr[0][i] ? (double) attr : boundsPerAttr[0][i];
					boundsPerAttr[1][i] = (double) attr > boundsPerAttr[1][i] ? (double) attr : boundsPerAttr[1][i];
				}
				i++;
			}
		}
		double[][] normalization = new double[2][numA];
		for (int i = 0; i < numA; i++) {
			if (this.instances.get(0).get(i) instanceof Double) {
				double diff = boundsPerAttr[1][i] - boundsPerAttr[0][i];
				normalization[0][i] =  diff == 0 ? Double.MAX_VALUE : 1 / diff;
				normalization[1][i] = - boundsPerAttr[0][i];
			}
		}
		return normalization;
	}

}
