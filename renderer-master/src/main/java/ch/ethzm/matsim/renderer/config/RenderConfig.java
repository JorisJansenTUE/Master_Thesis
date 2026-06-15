package ch.ethzm.matsim.renderer.config;

import java.io.File;
import java.util.ArrayList;
import java.util.LinkedList;
import java.util.List;

public class RenderConfig {
	public String outputPath;
	public String eventsPath;
	public String networkPath;
	public String backgroundImagePath = null;
	
	public double startTime = 0.0;
	public double endTime = 24 * 3600.0;
	public double secondsPerFrame = 120.0;

	public List<Double> center = new LinkedList<>();
	public double zoom = 10000.0;

	public int width = 1280;
	public int height = 720;

	public int binSize = 300;

	public List<NetworkConfig> networks = new ArrayList<>();
	public List<VehicleConfig> vehicles = new ArrayList<>();
	public List<ActivityConfig> activities = new ArrayList<>();

	public boolean showTime = false;
	public double minimumActivityDuration = 0.0;

	public void validate() {
		if (center.size() != 2) {
			throw new IllegalStateException("Center coordinate can only have two components");
		}

		if (!new File(eventsPath).exists()) {
			throw new IllegalStateException("Events path does not exist");
		}

		if (!new File(networkPath).exists()) {
			throw new IllegalStateException("Network path does not exist");
		}

		File outputFile = new File(outputPath);

		if (outputFile.exists() && !outputFile.isFile()) {
			throw new IllegalStateException("Output path is not a file");
		}

		networks.forEach(NetworkConfig::validate);
		vehicles.forEach(VehicleConfig::validate);
		activities.forEach(ActivityConfig::validate);
	}
}
