package thesis;

import com.google.inject.Singleton;

import org.matsim.contrib.bicycle.BicycleConfigGroup;
import org.matsim.contrib.bicycle.BicycleLinkSpeedCalculator;
import org.matsim.contrib.bicycle.BicycleLinkSpeedCalculatorDefaultImpl;
import org.matsim.contrib.bicycle.BicycleParams;
import org.matsim.contrib.bicycle.BicycleParamsDefaultImpl;
import org.matsim.contrib.bicycle.BicycleTravelDisutilityFactory;
import org.matsim.contrib.bicycle.BicycleTravelTime;
import org.matsim.core.config.ConfigUtils;
import org.matsim.core.controler.AbstractModule;

public class LocarnoBicycleRoutingModule extends AbstractModule {
    @Override
    public void install() {
        BicycleConfigGroup bicycleConfig = ConfigUtils.addOrGetModule(
                getConfig(),
                BicycleConfigGroup.class
        );

        // Needed by BicycleTravelTime and BicycleTravelDisutilityFactory
        bind(BicycleLinkSpeedCalculator.class)
                .to(BicycleLinkSpeedCalculatorDefaultImpl.class);

        bind(BicycleParams.class)
                .to(BicycleParamsDefaultImpl.class);

        // Only install routing components, not the QSim speed calculator.
        addTravelTimeBinding(bicycleConfig.getBicycleMode())
                .to(BicycleTravelTime.class)
                .in(Singleton.class);

        addTravelDisutilityFactoryBinding(bicycleConfig.getBicycleMode())
                .to(BicycleTravelDisutilityFactory.class)
                .in(Singleton.class);
    }
}