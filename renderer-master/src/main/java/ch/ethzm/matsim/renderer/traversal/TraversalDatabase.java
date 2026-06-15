package ch.ethzm.matsim.renderer.traversal;

import java.util.ArrayList;
import java.util.Collection;
import java.util.Collections;
import java.util.LinkedList;
import java.util.List;
import java.util.stream.Stream;

public class TraversalDatabase {
	final private List<List<Traversal>> traversals;

	final private double startTime;
	final private double binSize;
	final private int numberOfBins;

	public TraversalDatabase(double startTime, double endTime, double binSize) {
		this.startTime = startTime;
		this.binSize = binSize;

		this.numberOfBins = (int) Math.ceil(endTime - startTime / binSize);
		this.traversals = new ArrayList<>(numberOfBins);

		for (int i = 0; i < numberOfBins; i++) {
			traversals.add(new LinkedList<>());
		}
	}

	public void addTraversal(Traversal traversal) {
		int startBin = (int) Math.floor((traversal.startTime - startTime) / binSize);
		int endBin = (int) Math.floor((traversal.endTime - startTime) / binSize);

		if ((startBin >= 0 && startBin < numberOfBins) || (endBin >= 0 && endBin < numberOfBins)) {
			startBin = Math.max(0, startBin);
			endBin = Math.min(numberOfBins - 1, endBin);

			for (int bin = startBin; bin <= endBin; bin++) {
				traversals.get(bin).add(traversal);
			}
		}
	}

	public Collection<Traversal> getTraversalsForBin(double time) {
		int bin = (int) Math.floor((time - startTime) / binSize);
		return (bin >= 0 && bin < numberOfBins) ? traversals.get(bin) : Collections.emptyList();
	}

	public Stream<Traversal> getTraversalsAtTime(double time) {
		return getTraversalsForBin(time).parallelStream().filter(t -> t.startTime <= time && t.endTime > time);
	}

	public List<List<Traversal>> getData() {
		return traversals;
	}
}
