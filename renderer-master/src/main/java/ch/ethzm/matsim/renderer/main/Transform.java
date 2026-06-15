package ch.ethzm.matsim.renderer.main;

import java.awt.Rectangle;

import org.matsim.api.core.v01.Coord;

public class Transform {
	final private Rectangle windowBounds;
	final private Rectangle scenarioBounds;

	public Transform(Rectangle windowBounds, Rectangle scenarioBounds) {
		this.windowBounds = windowBounds;
		this.scenarioBounds = scenarioBounds;
	}

	public Coord windowToScenario(Coord windowCoord) {
		return new Coord(
				(windowCoord.getX() - windowBounds.getCenterX()) * scenarioBounds.getWidth() / windowBounds.getWidth()
						+ scenarioBounds.getCenterX(),
				(windowCoord.getY() - windowBounds.getCenterY()) * scenarioBounds.getHeight() / windowBounds.getHeight()
						+ scenarioBounds.getCenterY());
	}

	public Coord scenarioToWindow(Coord scenarioCoord) {
		return new Coord(
				(scenarioCoord.getX() - scenarioBounds.getCenterX()) * windowBounds.getWidth()
						/ scenarioBounds.getWidth() + windowBounds.getCenterX(),
				windowBounds.getHeight() - ((scenarioCoord.getY() - scenarioBounds.getCenterY()) * windowBounds.getHeight()
						/ scenarioBounds.getHeight() + windowBounds.getCenterY()));
	}

}
