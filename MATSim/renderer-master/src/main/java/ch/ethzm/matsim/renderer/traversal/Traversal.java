package ch.ethzm.matsim.renderer.traversal;

public class Traversal {
	final public double startTime;
	final public double endTime;
	final public int linkIndex;
	final public int vehicleIndex;
	final public int vehicleType;

	public Traversal(int linkIndex, int personIndex, int vehicleIndex, double startTime, double endTime,
			int vehicleType) {
		this.linkIndex = linkIndex;
		this.startTime = startTime;
		this.endTime = endTime;
		this.vehicleIndex = vehicleIndex;
		this.vehicleType = vehicleType;
	}
}
