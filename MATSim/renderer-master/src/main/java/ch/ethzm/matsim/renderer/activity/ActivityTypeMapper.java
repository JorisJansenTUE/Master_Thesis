package ch.ethzm.matsim.renderer.activity;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import ch.ethzm.matsim.renderer.config.ActivityConfig;
import ch.ethzm.matsim.renderer.config.RenderConfig;

public class ActivityTypeMapper {
	final private Map<String, Integer> map = new HashMap<>();
	final private List<String> types = new ArrayList<>();

	private int getType(String activityType) {
		int selectedIndex = 0;

		for (ActivityConfig activityConfig : renderConfig.activities) {
			if (activityConfig.isGeneric() || activityConfig.types.contains(activityType)) {
				return selectedIndex;
			}

			selectedIndex++;
		}

		return -1;
	}

	public int addActivityType(String activityType) {
		if (map.containsKey(activityType)) {
			return map.get(activityType);
		}

		int index = getType(activityType);
		map.put(activityType, index);
		types.add(activityType);

		return index;
	}

	public List<String> getData() {
		return types;
	}

	public String getActivityType(int index) {
		return types.get(index);
	}

	public boolean hasActivityType(String activityType) {
		return map.containsKey(activityType);
	}

	public int getIndex(String activityType) {
		Integer result = map.get(activityType);

		if (result == null) {
			throw new IllegalStateException(activityType);
		}

		return result;
	}

	private final RenderConfig renderConfig;

	public ActivityTypeMapper(RenderConfig renderConfig) {
		this.renderConfig = renderConfig;
	}
}
