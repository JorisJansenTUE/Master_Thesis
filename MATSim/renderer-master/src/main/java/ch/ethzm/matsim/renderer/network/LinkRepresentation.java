package ch.ethzm.matsim.renderer.network;

import org.matsim.api.core.v01.network.Link;

public class LinkRepresentation {
	public Link link;
	public int typeIndex;

	public LinkRepresentation(Link link, int typeIndex) {
		this.link = link;
		this.typeIndex = typeIndex;
	}
}
