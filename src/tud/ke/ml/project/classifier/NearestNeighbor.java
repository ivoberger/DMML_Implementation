package tud.ke.ml.project.classifier;

import sun.reflect.generics.reflectiveObjects.NotImplementedException;
import tud.ke.ml.project.util.Pair;

import java.io.Serializable;
import java.util.Comparator;
import java.util.List;
import java.util.Map;
import java.util.function.Function;
import java.util.stream.Collectors;

/**
 * This implementation assumes the class attribute is always available (but probably not set).
 */
public class NearestNeighbor extends INearestNeighbor implements Serializable {
	private static final long serialVersionUID = 1L;
	
	protected double[] scaling;
	protected double[] translation;
	
	private List<List<Object>> trainData;
	private int classAttribute;
	
	// TODO: add missing matrikel numbers
	@Override
	public String getMatrikelNumbers() {
		return "2857154,FELIX,LUKAS";
	}
	
	@Override
	protected void learnModel(List<List<Object>> data) {
		this.trainData = data;
	}
	
	@Override
	protected Map<Object, Double> getUnweightedVotes(List<Pair<List<Object>, Double>> subset) {
		return subset.stream()
				.map(entry -> entry.getA().get(getClassAttribute()))
				.collect(Collectors.groupingBy(Function.identity(), Collectors.counting()))
				.entrySet().stream().collect(Collectors.toMap(Map.Entry::getKey, value -> value.getValue().doubleValue()));
	}
	
	@Override
	protected Map<Object, Double> getWeightedVotes(List<Pair<List<Object>, Double>> subset) {
		throw new NotImplementedException();
	}
	
	@Override
	protected Object getWinner(Map<Object, Double> votes) {
		return votes.entrySet().stream().max(Map.Entry.comparingByValue()).get().getKey();
	}
	
	@Override
	protected Object vote(List<Pair<List<Object>, Double>> subset) {
		if (this.isInverseWeighting()) {
			return this.getWinner(this.getWeightedVotes(subset));
		}
		return this.getWinner(this.getUnweightedVotes(subset));
	}
	
	@Override
	protected List<Pair<List<Object>, Double>> getNearest(List<Object> data) {
		List<Pair<List<Object>, Double>> tmp = this.trainData.stream()
				.map(entry -> new Pair<>(entry, this.determineManhattanDistance(entry, data)))
				.sorted(Comparator.comparing(Pair::getB))
				.limit(this.getkNearest())
				.collect(Collectors.toList());
		//System.out.println(tmp);
		return tmp;
	}
	
	@Override
	protected double determineManhattanDistance(List<Object> instance1, List<Object> instance2) {
		int distance = 0;
		for (int i = 0; i < instance1.size(); i++) {
			if (i != getClassAttribute() && !instance1.get(i).equals(instance2.get(i)))
				distance++;
		}
		return distance;
	}
	
	@Override
	protected double determineEuclideanDistance(List<Object> instance1, List<Object> instance2) {
		throw new NotImplementedException();
	}
	
	@Override
	protected double[][] normalizationScaling() {
		throw new NotImplementedException();
	}
	
}
