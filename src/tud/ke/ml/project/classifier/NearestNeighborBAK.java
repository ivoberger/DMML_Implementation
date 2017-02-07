package tud.ke.ml.project.classifier;

import tud.ke.ml.project.util.Pair;
import weka.classifiers.lazy.keNN;

import java.io.Serializable;
import java.util.Arrays;
import java.util.Comparator;
import java.util.List;
import java.util.Map;
import java.util.function.Function;
import java.util.stream.Collectors;

/**
 * This implementation assumes the class attribute is always available (but probably not set).
 */
public class NearestNeighborBAK extends INearestNeighbor implements Serializable {
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
			this.scaling = normalization[0].clone();
			this.translation = normalization[1].clone();
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
				.map(entry -> new Pair<>(entry.getA().get(this.getClassAttribute()), 1 / (entry.getB()+0.001)))
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
			double[][] normalization = this.normalizationScaling();
			this.scaling = normalization[0].clone();
			this.translation = normalization[1].clone();
		} else {
			this.scaling = new double[this.numAttributes];
			this.translation = new double[this.numAttributes];
			Arrays.fill(this.scaling, 1);
			Arrays.fill(this.translation, 1);
		}

		switch (this.getMetric()) {
			case keNN.DIST_MANHATTAN:
				List<Pair<List<Object>, Double>> tmp = this.trainData.stream()
						.map(entry -> new Pair<>(entry, this.determineManhattanDistance(entry, data)))
						.sorted(Comparator.comparing(Pair::getB))
						.limit(this.getkNearest())
						.collect(Collectors.toList());
				return tmp;
			case keNN.DIST_EUCLIDEAN:
				tmp = this.trainData.stream()
						.map(entry -> new Pair<>(entry, this.determineEuclideanDistance(entry, data)))
						.sorted(Comparator.comparing(Pair::getB))
						.limit(this.getkNearest())
						.collect(Collectors.toList());
				return tmp;
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
				double a1 = this.isNormalizing() ? ((double) att1 + translation[i]) * this.scaling[i] : (double) att1;
				double a2 = this.isNormalizing() ? ((double) att2 + translation[i]) * this.scaling[i] : (double) att2;
				distance += Math.abs(a1 - a2);
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
				double a1 = this.isNormalizing() ? ((double) att1 + translation[i]) * this.scaling[i] : (double) att1;
				double a2 = this.isNormalizing() ? ((double) att2 + translation[i]) * this.scaling[i] : (double) att2;
				distance += Math.pow(Math.abs(a1 - a2), 2);
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
			if (this.trainData.get(0).get(i) instanceof Double) {
				double diff = boundsPerAttr[1][i] - boundsPerAttr[0][i];
				normalization[0][i] =  diff == 0 ? Double.MAX_VALUE : 1 / diff;
				normalization[1][i] = - boundsPerAttr[0][i];
			}
		}
		return normalization;
	}
	
}
