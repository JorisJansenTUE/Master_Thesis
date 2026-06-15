package ch.ethzm.matsim.renderer.traversal;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import org.matsim.api.core.v01.Id;
import org.matsim.vehicles.Vehicle;

import ch.ethzm.matsim.renderer.config.RenderConfig;
import ch.ethzm.matsim.renderer.config.VehicleConfig;

public class VehicleDatabase {
	final private Map<String, Integer> map = new HashMap<>();
	final private List<String> ids = new ArrayList<>();
	final private List<Integer> types = new ArrayList<>(10000);

	public int addVehicle(Id<Vehicle> vehicleId) {
		return addVehicle(vehicleId.toString());
	}

	public int addVehicle(String vehicleId) {
		if (map.containsKey(vehicleId)) {
			return map.get(vehicleId);
		} else {
			int index = map.size();
			map.put(vehicleId, index);

			int typeIndex = -1;
			int selectedIndex = 0;

			for (VehicleConfig vehicleConfig : renderConfig.vehicles) {
				boolean isRelevant = false;

				if (vehicleConfig.isGeneric()) {
					isRelevant = true;
				}

				for (String prefix : vehicleConfig.startsWith) {
					if (vehicleId.startsWith(prefix)) {
						isRelevant = true;
					}
				}

				for (String suffix : vehicleConfig.endsWith) {
					if (vehicleId.startsWith(suffix)) {
						isRelevant = true;
					}
				}

				for (String part : vehicleConfig.contains) {
					if (vehicleId.contains(part)) {
						isRelevant = true;
					}
				}

				if (isRelevant) {
					typeIndex = selectedIndex;
				}

				selectedIndex++;
			}

			types.add(typeIndex);

			return index;
		}
	}

	public Map<String, Integer> getVehicleIndices() {
		return map;
	}

	public List<String> getData() {
		return ids;
	}

	public int getType(int vehicleIndex) {
		return types.get(vehicleIndex);
	}

	private final RenderConfig renderConfig;

	public VehicleDatabase(RenderConfig renderConfig) {
		this.renderConfig = renderConfig;
	}
}
