package ch.ethzm.matsim.renderer.main;

import org.matsim.api.core.v01.network.Link;
import org.matsim.api.core.v01.network.Network;
import org.matsim.core.api.experimental.events.EventsManager;
import org.matsim.core.events.EventsManagerImpl;
import org.matsim.core.events.MatsimEventsReader;
import org.matsim.core.network.NetworkUtils;
import org.matsim.core.network.io.MatsimNetworkReader;

import ch.ethzm.matsim.renderer.activity.ActivityDatabase;
import ch.ethzm.matsim.renderer.activity.ActivitySliceListener;
import ch.ethzm.matsim.renderer.activity.ActivityTypeMapper;
import ch.ethzm.matsim.renderer.config.RenderConfig;
import ch.ethzm.matsim.renderer.network.LinkDatabase;
import ch.ethzm.matsim.renderer.renderer.Renderer;
import ch.ethzm.matsim.renderer.renderer.VideoTarget;
import ch.ethzm.matsim.renderer.traversal.TraversalDatabase;
import ch.ethzm.matsim.renderer.traversal.TraversalListener;
import ch.ethzm.matsim.renderer.traversal.VehicleDatabase;

public class RunRenderer {
	static public void run(RenderConfig renderConfig) {
		renderConfig.validate();

		// Prepare databases to read information

		double startTime = renderConfig.startTime;
		double endTime = renderConfig.endTime;
		double binSize = renderConfig.binSize;

		TraversalDatabase traversalDatabase = new TraversalDatabase(startTime, endTime, binSize);
		ActivityDatabase activityDatabase = new ActivityDatabase(startTime, endTime, binSize);

		LinkDatabase linkDatabase = new LinkDatabase(renderConfig);
		ActivityTypeMapper activityTypeMapper = new ActivityTypeMapper(renderConfig);

		VehicleDatabase vehicleDatabase = new VehicleDatabase(renderConfig);

		// Read input data

		String networkPath = renderConfig.networkPath;
		String eventsPath = renderConfig.eventsPath;

		Network network = NetworkUtils.createNetwork();
		new MatsimNetworkReader(network).readFile(networkPath);

		for (Link link : network.getLinks().values()) {
			linkDatabase.addLink(link);
		}

		EventsManager eventsManager = new EventsManagerImpl();
		TraversalListener traversalListener = new TraversalListener(linkDatabase, traversalDatabase, network,
				vehicleDatabase);
		ActivitySliceListener activityListener = new ActivitySliceListener(linkDatabase, activityDatabase,
				activityTypeMapper, network, renderConfig.minimumActivityDuration);
		eventsManager.addHandler(traversalListener);
		eventsManager.addHandler(activityListener);
		new MatsimEventsReader(eventsManager).readFile(eventsPath);

		Renderer renderer = new Renderer(renderConfig.width, renderConfig.height, renderConfig, traversalDatabase,
				linkDatabase, activityDatabase, activityTypeMapper, vehicleDatabase);

		new VideoTarget(renderConfig).run(renderer);
	}
}
