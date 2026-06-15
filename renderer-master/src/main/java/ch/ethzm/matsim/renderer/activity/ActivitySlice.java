package ch.ethzm.matsim.renderer.activity;

public class ActivitySlice {
	final public double startTime;
	final public double endTime;
	final public int typeIndex;
	final public int linkIndex;

	public ActivitySlice(double startTime, double endTime, int typeIndex, int linkIndex) {
		this.startTime = startTime;
		this.endTime = endTime;
		this.typeIndex = typeIndex;
		this.linkIndex = linkIndex;
	}
}
