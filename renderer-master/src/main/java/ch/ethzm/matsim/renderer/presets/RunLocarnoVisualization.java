package ch.ethzm.matsim.renderer.presets;

import java.util.Arrays;

import ch.ethzm.matsim.renderer.config.ActivityConfig;
import ch.ethzm.matsim.renderer.config.NetworkConfig;
import ch.ethzm.matsim.renderer.config.RenderConfig;
import ch.ethzm.matsim.renderer.config.VehicleConfig;
import ch.ethzm.matsim.renderer.main.RunRenderer;

public class RunLocarnoVisualization {
	static public void main(String[] args) {
		// START CONFIGURATION

		RenderConfig renderConfig = new RenderConfig();

		renderConfig.width = 1280;
		renderConfig.height = 720;

		renderConfig.networkPath = "../MATSim/output/run617/output_network.xml.gz";
		renderConfig.eventsPath = "../MATSim/output/run617/output_events.xml.gz";
		renderConfig.outputPath = "../MATSim/output/run617/animation.mp4";
		renderConfig.backgroundImagePath = "C:/Users/20201733/Downloads/animation_background_1506.png";
        
		renderConfig.startTime = 8.0 * 3600.0;
		renderConfig.endTime =  10 * 3600.0;
		renderConfig.secondsPerFrame = 60.0;

		renderConfig.showTime = true;

        renderConfig.center = Arrays.asList(2705000.0, 1113500.0);
        renderConfig.zoom = 6000.0;

		
        NetworkConfig bikeNetwork = new NetworkConfig();
        renderConfig.networks.add(bikeNetwork);
        bikeNetwork.modes = Arrays.asList("bike");
        bikeNetwork.color = Arrays.asList(220,83,30);

        NetworkConfig carNetwork = new NetworkConfig();
        renderConfig.networks.add(carNetwork);
        carNetwork.modes = Arrays.asList("car");
        carNetwork.color = Arrays.asList(160, 160, 160);

		NetworkConfig subwayNetwork = new NetworkConfig();
		renderConfig.networks.add(subwayNetwork);
		subwayNetwork.modes = Arrays.asList("rail");
		subwayNetwork.color = Arrays.asList(225, 127, 245);

        VehicleConfig defaultVehicle = new VehicleConfig();
        renderConfig.vehicles.add(defaultVehicle); 
        defaultVehicle.color = Arrays.asList(0, 114, 189);
        defaultVehicle.size = 6;

		VehicleConfig busVehicle = new VehicleConfig();
		renderConfig.vehicles.add(busVehicle);
        busVehicle.contains = Arrays.asList("bus");
		busVehicle.color = Arrays.asList(128,46,144);
		busVehicle.size = 6;

        VehicleConfig bikeVehicle = new VehicleConfig();
        renderConfig.vehicles.add(bikeVehicle);
        bikeVehicle.contains = Arrays.asList("bike");
        bikeVehicle.color = Arrays.asList(220,83,30);
        bikeVehicle.size = 6;

		VehicleConfig ptVehicle = new VehicleConfig();
		renderConfig.vehicles.add(ptVehicle);
		ptVehicle.contains = Arrays.asList("rail");
		ptVehicle.color = Arrays.asList(225, 127, 245); // .asList(7, 145, 222);
		ptVehicle.size = 6;

		ActivityConfig workActivity = new ActivityConfig();
		renderConfig.activities.add(workActivity);
		workActivity.types.add("work");
		workActivity.maximumLifetime = 300.0;
		workActivity.size = 16;
		workActivity.color = Arrays.asList(85,222,27);

		// END CONFIGURATION

		RunRenderer.run(renderConfig);
	}
}

