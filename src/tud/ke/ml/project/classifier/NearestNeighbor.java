package tud.ke.ml.project.classifier;

import tud.ke.ml.project.util.Pair;
import weka.classifiers.lazy.keNN;

import java.io.Serializable;
import java.util.Comparator;
import java.util.HashMap;
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
		return "2857154,2750840,2623168";
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
				.map(entry -> new Pair<>(entry.getA().get(this.getClassAttribute()), 1 / (entry.getB() + 0.001)))
				.collect(Collectors.groupingBy(Pair::getA, Collectors.summingDouble(Pair::getB)))
				.entrySet().stream().collect(Collectors.toMap(Map.Entry::getKey, Map.Entry::getValue));
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
		List<Pair<List<Object>, Double>> results;
		switch (this.getMetric()) {
			case keNN.DIST_MANHATTAN:
				results = this.trainData.stream()
						.map(entry -> new Pair<>(entry, this.determineManhattanDistance(entry, data)))
						.sorted(Comparator.comparing(Pair::getB))
						.collect(Collectors.toList());
				break;
			case keNN.DIST_EUCLIDEAN:
				results = this.trainData.stream()
						.map(entry -> new Pair<>(entry, this.determineEuclideanDistance(entry, data)))
						.sorted(Comparator.comparing(Pair::getB))
						.collect(Collectors.toList());
				break;
			default:
				throw new UnknownError("Metric unknown");
		}
		return this.filterNeighbours(results);
	}
	
	private List<Pair<List<Object>, Double>> filterNeighbours(List<Pair<List<Object>, Double>> data) {
		if (this.getkNearest() == 1) return data.stream().limit(this.getkNearest()).collect(Collectors.toList());
		
		int equals = 0;
		boolean foundEquals = false;
		HashMap<String, Integer> classes = new HashMap<>();
		for (int i = 0; i < data.size() - 1; i++) {
			double dist1 = data.get(i).getB().doubleValue();
			double dist2 = data.get(i + 1).getB().doubleValue();
			if (foundEquals && dist1 != dist2) {
				foundEquals = false;
			}
			if (!foundEquals && dist1 == dist2) {
				equals = i;
				foundEquals = true;
			}
			if (i >= this.getkNearest() && !foundEquals) break;
			
			String currentClass = data.get(i).getA().toString();
			classes.put(currentClass, classes.getOrDefault(currentClass, 0) + 1);
		}
		if (!foundEquals) return data.stream().limit(this.getkNearest()).collect(Collectors.toList());
		HashMap<String, Integer> classesOrdered = new HashMap<>();
		classes.entrySet().stream().sorted(Map.Entry.comparingByValue()).forEachOrdered(x -> classesOrdered.put(x.getKey(), x.getValue()));
		
		List<Pair<List<Object>, Double>> result = data.stream().limit(this.getkNearest() - equals).collect(Collectors.toList());
		for (String key : classesOrdered.keySet()) {
			for (int i = equals; i < data.size(); i++) {
				String currentClass = data.get(i).getA().toString();
				if (currentClass.equals(key) && result.size() <= this.getkNearest()) {
					result.add(data.get(i));
				} else break;
			}
		}
		return result;
	}
	
	@Override
	protected double determineManhattanDistance(List<Object> instance1, List<Object> instance2) {
		double distance = 0;
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
		double distance = 0;
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
		for (int i = 0; i < this.trainData.size(); i++) {
			for (int j = 0; j < this.numAttributes; j++) {
				Object attr = this.trainData.get(i).get(j);
				if (attr instanceof Double) {
					double dAttr = (double) attr;
					if (i == 0) {
						boundsPerAttr[0][j] = dAttr;
						boundsPerAttr[1][j] = dAttr;
					}
					boundsPerAttr[0][j] = dAttr < boundsPerAttr[0][j] ? dAttr : boundsPerAttr[0][j];
					boundsPerAttr[1][j] = dAttr > boundsPerAttr[1][j] ? dAttr : boundsPerAttr[1][j];
				}
			}
		}
		
		double[][] normalization = new double[2][this.numAttributes];
		for (int i = 0; i < this.numAttributes; i++) {
			if (this.trainData.get(0).get(i) instanceof Double) {
				double diff = boundsPerAttr[1][i] - boundsPerAttr[0][i];
				normalization[0][i] = diff == 0 ? Double.MAX_VALUE : 1 / diff;
				normalization[1][i] = -boundsPerAttr[0][i];
			}
		}
		return normalization;
	}
	
}
