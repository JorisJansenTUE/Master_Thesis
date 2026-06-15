package ch.ethzm.matsim.renderer.activity;

import java.util.ArrayList;
import java.util.Collection;
import java.util.Collections;
import java.util.List;
import java.util.stream.Stream;

public class ActivityDatabase {
	final private List<List<ActivitySlice>> activities;

	final private double startTime;
	final private double binSize;
	final private int numberOfBins;

	public ActivityDatabase(double startTime, double endTime, double binSize) {
		this.startTime = startTime;
		this.binSize = binSize;

		this.numberOfBins = (int) Math.ceil(endTime - startTime / binSize);
		this.activities = new ArrayList<>(numberOfBins);

		for (int i = 0; i < numberOfBins; i++) {
			activities.add(new ArrayList<>());
		}
	}

	public void addActivity(ActivitySlice traversal) {
		int startBin = (int) Math.floor((traversal.startTime - startTime) / binSize);
		int endBin = (int) Math.floor((traversal.endTime - startTime) / binSize);

		if ((startBin >= 0 && startBin < numberOfBins) || (endBin >= 0 && endBin < numberOfBins)) {
			startBin = Math.max(0, startBin);
			endBin = Math.min(numberOfBins - 1, endBin);

			for (int bin = startBin; bin <= endBin; bin++) {
				activities.get(bin).add(traversal);
			}
		}
	}

	public Collection<ActivitySlice> getActivitiesForBin(double time) {
		int bin = (int) Math.floor((time - startTime) / binSize);
		return (bin >= 0 && bin < numberOfBins) ? activities.get(bin) : Collections.emptyList();
	}

	public Stream<ActivitySlice> getActivitiesAtTime(double time) {
		return getActivitiesForBin(time).parallelStream().filter(t -> t.startTime <= time && t.endTime > time);
	}

	public List<List<ActivitySlice>> getData() {
		return activities;
	}
}
