package ch.ethzm.matsim.renderer.network;

import java.util.ArrayList;
import java.util.Collection;
import java.util.List;

import org.matsim.api.core.v01.network.Link;

import ch.ethzm.matsim.renderer.config.NetworkConfig;
import ch.ethzm.matsim.renderer.config.RenderConfig;

public class LinkDatabase {
	private final ArrayList<LinkRepresentation> links = new ArrayList<>(100000);
	private final List<NetworkConfig> networks;

	public LinkDatabase(RenderConfig renderConfig) {
		this.networks = renderConfig.networks;
	}

	public int addLink(Link link) {
		links.ensureCapacity(link.getId().index() + 10000);

		while (links.size() <= link.getId().index()) {
			links.add(null);
		}

		int typeIndex = -1;

		int selectedIndex = 0;

		for (NetworkConfig network : networks) {
			for (String mode : network.modes) {
				if (link.getAllowedModes().contains(mode)) {
					typeIndex = selectedIndex;
				}
			}

			selectedIndex++;
		}

		links.set(link.getId().index(), new LinkRepresentation(link, typeIndex));
		return link.getId().index();
	}

	public LinkRepresentation getLink(int index) {
		return links.get(index);
	}

	public int getSize() {
		return links.size();
	}

	public int getIndex(Link link) {
		return link.getId().index();
	}

	public Collection<LinkRepresentation> getLinks() {
		return links;
	}
}
