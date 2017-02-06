package tud.ke.ml.project.classifier;

import tud.ke.ml.project.util.Pair;
import weka.classifiers.lazy.keNN;

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
	private int numAttributes;
	
	// TODO: add missing matrikel numbers
	@Override
	public String getMatrikelNumbers() {
		return "2857154,FELIX,LUKAS";
	}
	
	@Override
	protected void learnModel(List<List<Object>> data) {
		this.trainData = data;
		this.numAttributes = data.get(0).size();
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
		return subset.stream()
				.map(entry -> new Pair<>(entry.getA().get(this.getClassAttribute()), 1 / entry.getB()))
				.collect(Collectors.groupingBy(Pair::getA, Collectors.summingDouble(Pair::getB)))
				.entrySet().stream().collect(Collectors.toMap(Map.Entry::getKey, value -> value.getValue()));
	}
	
	@Override
	protected Object getWinner(Map<Object, Double> votes) {
		return votes.entrySet().stream().max(Map.Entry.comparingByValue()).get().getKey();
	}
	
	@Override
	protected Object vote(List<Pair<List<Object>, Double>> subset) {
		return this.isInverseWeighting() ? this.getWinner(this.getWeightedVotes(subset)) : this.getWinner(this.getUnweightedVotes(subset));
	}
	
	@Override
	protected List<Pair<List<Object>, Double>> getNearest(List<Object> data) {
		if (this.isNormalizing()) {
			this.scaling = this.normalizationScaling()[0];
			this.translation = this.normalizationScaling()[1];
		}
		
		switch (this.getMetric()) {
			case keNN.DIST_MANHATTAN:
				return this.trainData.stream()
						.map(entry -> new Pair<>(entry, this.determineManhattanDistance(entry, data)))
						.sorted(Comparator.comparing(Pair::getB))
						.limit(this.getkNearest())
						.collect(Collectors.toList());
			case keNN.DIST_EUCLIDEAN:
				return this.trainData.stream()
						.map(entry -> new Pair<>(entry, this.determineEuclideanDistance(entry, data)))
						.sorted(Comparator.comparing(Pair::getB))
						.limit(this.getkNearest())
						.collect(Collectors.toList());
		}
		throw new UnknownError("Metric unknown");
	}
	
	@Override
	protected double determineManhattanDistance(List<Object> instance1, List<Object> instance2) {
		int distance = 0;
		for (int i = 0; i < this.numAttributes; i++) {
			if (i == this.getClassAttribute()) continue;
			Object att1 = instance1.get(i);
			Object att2 = instance2.get(i);
			if (att1 instanceof String) {
				if (!att1.equals(att2)) {
					distance++;
				}
			} else if (att1 instanceof Double) {
				distance += Math.abs((Double) att1 - (Double) att2);
			}
		}
		return distance;
	}
	
	@Override
	protected double determineEuclideanDistance(List<Object> instance1, List<Object> instance2) {
		int distance = 0;
		for (int i = 0; i < this.numAttributes; i++) {
			if (i == this.getClassAttribute()) continue;
			Object att1 = instance1.get(i);
			Object att2 = instance2.get(i);
			if (att1 instanceof String) {
				if (!att1.equals(att2)) {
					distance++;
				}
			} else if (att1 instanceof Double) {
				distance += Math.pow(Math.abs((Double) att1 - (Double) att2), 2);
			}
		}
		return Math.sqrt(distance);
	}
	
	@Override
	protected double[][] normalizationScaling() {
		double[][] boundsPerAttr = new double[this.numAttributes][2];
		System.out.println(this.numAttributes);
		for (List<Object> instance : this.trainData) {
			int i = 0;
			for (Object attr : instance) {
				if (attr instanceof Double) {
					boundsPerAttr[i][0] = (double) attr < boundsPerAttr[i][0] ? (double) attr : boundsPerAttr[i][0];
					boundsPerAttr[i][1] = (double) attr > boundsPerAttr[i][1] ? (double) attr : boundsPerAttr[i][1];
				} else {
					boundsPerAttr[i][0] = -1;
					boundsPerAttr[i][1] = -2;
				}
				i++;
			}
		}
		for (double bound[] : boundsPerAttr) {
			System.out.println("Lower Bound: " + bound[0]);
			System.out.println("Upper Bound: " + bound[1]);
			System.out.println();
		}
		
		return boundsPerAttr;
	}
	
}
