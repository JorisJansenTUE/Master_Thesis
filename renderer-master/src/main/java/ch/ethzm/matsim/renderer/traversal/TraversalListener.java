package ch.ethzm.matsim.renderer.traversal;

import java.util.HashMap;
import java.util.Map;

import org.matsim.api.core.v01.Id;
import org.matsim.api.core.v01.events.LinkEnterEvent;
import org.matsim.api.core.v01.events.LinkLeaveEvent;
import org.matsim.api.core.v01.events.VehicleLeavesTrafficEvent;
import org.matsim.api.core.v01.events.handler.LinkEnterEventHandler;
import org.matsim.api.core.v01.events.handler.LinkLeaveEventHandler;
import org.matsim.api.core.v01.events.handler.VehicleLeavesTrafficEventHandler;
import org.matsim.api.core.v01.network.Network;
import org.matsim.vehicles.Vehicle;

import ch.ethzm.matsim.renderer.network.LinkDatabase;

public class TraversalListener
		implements LinkEnterEventHandler, LinkLeaveEventHandler, VehicleLeavesTrafficEventHandler {
	final private LinkDatabase linkDatabase;
	final private TraversalDatabase traversalDatabase;
	final private Network network;
	final private VehicleDatabase vehicleDatabase;

	final private Map<Id<Vehicle>, LinkEnterEvent> enterEvents = new HashMap<>();

	public TraversalListener(LinkDatabase linkDatabase, TraversalDatabase traversalDatabase, Network network,
			VehicleDatabase vehicleDatabase) {
		this.linkDatabase = linkDatabase;
		this.traversalDatabase = traversalDatabase;
		this.network = network;
		this.vehicleDatabase = vehicleDatabase;
	}

	@Override
	public void handleEvent(LinkEnterEvent event) {
		enterEvents.put(event.getVehicleId(), event);
	}

	@Override
	public void handleEvent(LinkLeaveEvent leaveEvent) {
		LinkEnterEvent enterEvent = enterEvents.get(leaveEvent.getVehicleId());

		if (enterEvent != null) {
			int linkIndex = enterEvent.getLinkId().index();
			int vehicleIndex = vehicleDatabase.addVehicle(enterEvent.getVehicleId());
			int vehicleType = vehicleDatabase.getType(vehicleIndex);

			Traversal traversal = new Traversal(linkIndex, -1, vehicleIndex, enterEvent.getTime(), leaveEvent.getTime(),
					vehicleType);
			traversalDatabase.addTraversal(traversal);
		}
	}

	@Override
	public void handleEvent(VehicleLeavesTrafficEvent event) {
		enterEvents.remove(event.getVehicleId());
	}
}
