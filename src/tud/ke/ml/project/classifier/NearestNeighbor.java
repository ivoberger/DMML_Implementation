package tud.ke.ml.project.classifier;

import tud.ke.ml.project.util.Pair;
import weka.classifiers.lazy.keNN;

import java.io.Serializable;
import java.util.Comparator;
import java.util.LinkedList;
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
		
		if (this.isNormalizing()) {
			double[][] normalization = this.normalizationScaling();
			this.scaling = normalization[0];
			this.translation = normalization[1];
		}
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
		List<Object> inst1 = this.isNormalizing() ? this.normalize(instance1) : instance1;
		List<Object> inst2 = this.isNormalizing() ? this.normalize(instance2) : instance2;
		
		int distance = 0;
		for (int i = 0; i < this.numAttributes; i++) {
			if (i == this.getClassAttribute()) continue;
			Object att1 = inst1.get(i);
			Object att2 = inst2.get(i);
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
		List<Object> inst1 = this.isNormalizing() ? this.normalize(instance1) : instance1;
		List<Object> inst2 = this.isNormalizing() ? this.normalize(instance2) : instance2;
		
		int distance = 0;
		for (int i = 0; i < this.numAttributes; i++) {
			if (i == this.getClassAttribute()) continue;
			Object att1 = inst1.get(i);
			Object att2 = inst2.get(i);
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
		// save minimum and maximum per attribute
		double[][] boundsPerAttr = new double[2][this.numAttributes];
		for (List<Object> instance : this.trainData) {
			int i = 0;
			for (Object attr : instance) {
				if (attr instanceof Double) {
					boundsPerAttr[0][i] = (double) attr < boundsPerAttr[0][i] ? (double) attr : boundsPerAttr[0][i];
					boundsPerAttr[1][i] = (double) attr > boundsPerAttr[1][i] ? (double) attr : boundsPerAttr[1][i];
				}
				i++;
			}
		}
		double[][] normalization = new double[2][this.numAttributes];
		for (int i = 0; i < this.numAttributes; i++) {
			double diff = boundsPerAttr[1][i] - boundsPerAttr[0][i] == 0 ? 1 : boundsPerAttr[1][i] - boundsPerAttr[0][i];
			normalization[0][i] = 1 / diff;
			normalization[1][i] = -boundsPerAttr[0][i];
		}
		return normalization;
	}
	
	private List<Object> normalize(List<Object> instance) {
		List<Object> copy = new LinkedList<Object>(instance);
		for (int i = 0; i < this.numAttributes; i++) {
			if (copy.get(i) instanceof Double) {
				if ((double) copy.get(i) < 0 || (double) copy.get(i) > 1) {
					System.out.print("Before: " + copy.get(i));
					copy.set(i, (double) copy.get(i) * this.scaling[i]);
					copy.set(i, (double) copy.get(i) + this.translation[i]);
					System.out.print("  After: " + copy.get(i));
					System.out.print("  Scaling: " + this.scaling[i]);
					System.out.println("  Translation: " + this.translation[i]);
				}
				if ((double) copy.get(i) < 0 || (double) copy.get(i) > 1) {
					throw new UnknownError("Normalization screwed");
				}
			}
		}
		return copy;
	}
	
}
