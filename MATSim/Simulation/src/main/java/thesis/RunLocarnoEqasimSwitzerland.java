package thesis;

import java.nio.file.Path;
import java.nio.file.Paths;

import org.eqasim.switzerland.ch.SwitzerlandConfigurator;
import org.matsim.api.core.v01.Scenario;
import org.matsim.api.core.v01.population.Person;
import org.matsim.core.config.CommandLine;
import org.matsim.core.config.Config;
import org.matsim.core.config.ConfigGroup;
import org.matsim.core.config.ConfigUtils;
import org.matsim.core.config.groups.QSimConfigGroup;
import org.matsim.core.controler.Controler;
import org.matsim.core.population.PersonUtils;
import org.matsim.core.scenario.ScenarioUtils;

import ch.sbb.matsim.config.SwissRailRaptorConfigGroup;
import org.eqasim.core.components.config.EqasimConfigGroup;
import org.eqasim.core.components.raptor.EqasimRaptorConfigGroup;
import org.matsim.contribs.discrete_mode_choice.modules.config.DiscreteModeChoiceConfigGroup;


public class RunLocarnoEqasimSwitzerland {
    public static void main(String[] args) throws CommandLine.ConfigurationException {
        CommandLine cmd = new CommandLine.Builder(args)
                .requireOptions("config-path")
                .allowOptions("preventwaitingtoentertraffic")
                .build();

        if (!cmd.hasOption("config-path")) {
            throw new IllegalArgumentException(
                    "Usage: RunLocarnoEqasimSwitzerland --config-path <path-to-config.xml>"
            );
        }

        Path configPath = Paths.get(cmd.getOptionStrict("config-path")).toAbsolutePath().normalize();

        Path scenarioDirectory = configPath.getParent();

        SwitzerlandConfigurator configurator = new SwitzerlandConfigurator(cmd);

        Config config = ConfigUtils.loadConfig(
                cmd.getOptionStrict("config-path"),
                new EqasimConfigGroup(),
                new DiscreteModeChoiceConfigGroup(),
                new SwissRailRaptorConfigGroup(),
                new EqasimRaptorConfigGroup()
        );

        configurator.updateConfig(config);
        cmd.applyConfiguration(config);

        resolveOutputDirectoryRelativeToConfig(config, scenarioDirectory);

        if ("y".equals(cmd.getOption("preventwaitingtoentertraffic").orElse("n"))) {
            config.qsim().setPcuThresholdForFlowCapacityEasing(1.0);
        }

        Scenario scenario = ScenarioUtils.createScenario(config);
        configurator.configureScenario(scenario);

        ScenarioUtils.loadScenario(scenario);

        configurator.adjustScenario(scenario);
        ensureSwissModeChoiceAttributes(scenario);

        configurator.adjustPTpcu(scenario);

        Controler controller = new Controler(scenario);
        configurator.configureController(controller);

        controller.run();
    }

    private static void resolveOutputDirectoryRelativeToConfig(Config config, Path scenarioDirectory) {
        Path outputDirectory = Paths.get(config.controller().getOutputDirectory());

        if (!outputDirectory.isAbsolute()) {
            Path resolvedOutputDirectory = scenarioDirectory
                    .resolve(outputDirectory)
                    .normalize();

            config.controller().setOutputDirectory(resolvedOutputDirectory.toString());

            System.out.println(
                    "Resolved MATSim output directory to: "
                            + resolvedOutputDirectory
            );
        }
    }


    private static void ensureSwissModeChoiceAttributes(Scenario scenario) {
        for (Person person : scenario.getPopulation().getPersons().values()) {
            Object hasLicense = person.getAttributes().getAttribute("hasLicense");
            Object hasBike = person.getAttributes().getAttribute("hasBike");
            Object carAvail = person.getAttributes().getAttribute("carAvail");
            Object ptSubscription = person.getAttributes().getAttribute("ptSubscription");

            // MATSim PersonUtils.getLicense(person) expects the attribute "hasLicense"
            // as a String such as "yes" or "no", not necessarily as a Boolean.
            if (!(hasLicense instanceof String)) {
                boolean licenseAvailable = false;

                if (hasLicense instanceof Boolean) {
                    licenseAvailable = (Boolean) hasLicense;
                } else if (hasLicense != null) {
                    String value = hasLicense.toString().toLowerCase();
                    licenseAvailable = value.equals("true") || value.equals("yes");
                }

                person.getAttributes().putAttribute(
                        "hasLicense",
                        licenseAvailable ? "yes" : "no"
                );
            }

            // SwissModeAvailability calls PersonUtils.getCarAvail(person).equals("never"),
            // so carAvail must not be missing.
            if (carAvail == null) {
                person.getAttributes().putAttribute("carAvail", "never");
            }

            // SwissModeAvailability directly checks bikeAvailability.equals("FOR_NONE"),
            // so bikeAvailability must not be missing.
            if (person.getAttributes().getAttribute("bikeAvailability") == null) {
                boolean bikeAvailable = false;

                if (hasBike instanceof Boolean) {
                    bikeAvailable = (Boolean) hasBike;
                } else if (hasBike != null) {
                    String value = hasBike.toString().toLowerCase();
                    bikeAvailable = value.equals("true") || value.equals("yes");
                }

                person.getAttributes().putAttribute(
                        "bikeAvailability",
                        bikeAvailable ? "FOR_ALL" : "FOR_NONE"
                );
            }

            // Public transport subscriptions.
            boolean hasPtSubscription = false;
            if (ptSubscription instanceof Boolean) {
                hasPtSubscription = (Boolean) ptSubscription;
            } else if (ptSubscription != null) {
                String value = ptSubscription.toString().toLowerCase();
                hasPtSubscription = value.equals("true") || value.equals("yes");
            }

            if (person.getAttributes().getAttribute("ptHasGA") == null) {
                person.getAttributes().putAttribute("ptHasGA", false);
            }

            if (person.getAttributes().getAttribute("ptHasHalbtax") == null) {
                person.getAttributes().putAttribute("ptHasHalbtax", hasPtSubscription);
            }

            if (person.getAttributes().getAttribute("ptHasVerbund") == null) {
                person.getAttributes().putAttribute("ptHasVerbund", false);
            }

            if (person.getAttributes().getAttribute("spRegion") == null) {
                person.getAttributes().putAttribute("spRegion", -1);
            }
        }
    }


    // private static void ensureSwissModeChoiceAttributes(Scenario scenario) {
    //     /*To be used later when Demand Pipeline is updated, for now use above version */
    //     for (Person person : scenario.getPopulation().getPersons().values()) {
    //         if (PersonUtils.getLicense(person) == null) {
    //             PersonUtils.setLicense(person, "yes");
    //         }

    //         if (PersonUtils.getCarAvail(person) == null) {
    //             PersonUtils.setCarAvail(person, "always");
    //         }

    //         if (person.getAttributes().getAttribute("bikeAvailability") == null) {
    //             person.getAttributes().putAttribute("bikeAvailability", "FOR_ALL");
    //         }

    //         if (person.getAttributes().getAttribute("ptHasGA") == null) {
    //             person.getAttributes().putAttribute("ptHasGA", false);
    //         }

    //         if (person.getAttributes().getAttribute("ptHasHalbtax") == null) {
    //             person.getAttributes().putAttribute("ptHasHalbtax", false);
    //         }

    //         if (person.getAttributes().getAttribute("ptHasVerbund") == null) {
    //             person.getAttributes().putAttribute("ptHasVerbund", false);
    //         }

    //         if (person.getAttributes().getAttribute("spRegion") == null) {
    //             person.getAttributes().putAttribute("spRegion", -1);
    //         }
    //     }
    // }
}