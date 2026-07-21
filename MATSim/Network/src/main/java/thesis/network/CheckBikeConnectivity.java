package thesis.network;

import java.util.ArrayDeque;
import java.util.HashSet;
import java.util.Queue;
import java.util.Set;

import org.matsim.api.core.v01.Id;
import org.matsim.api.core.v01.TransportMode;
import org.matsim.api.core.v01.network.Link;
import org.matsim.api.core.v01.network.Network;
import org.matsim.api.core.v01.network.Node;
import org.matsim.core.network.NetworkUtils;
import org.matsim.core.network.io.MatsimNetworkReader;

public class CheckBikeConnectivity {

    public static void main(String[] args) {
        if (args.length != 3) {
            throw new IllegalArgumentException(
                "Usage: <network-file> <start-link-id> <end-link-id>"
            );
        }

        Network network = NetworkUtils.createNetwork();
        new MatsimNetworkReader(network).readFile(args[0]);
        long restrictions = network.getLinks().values().stream()
            .filter(link -> link.getAllowedModes().contains("bike"))
            .filter(link -> NetworkUtils.getDisallowedNextLinks(link) != null)
            .peek(link -> System.out.println(
                link.getId() + ": "
                    + NetworkUtils.getDisallowedNextLinks(link)
            ))
            .count();

        System.out.println("Bike links with turn restrictions: " + restrictions);
        
        Link startLink = network.getLinks().get(Id.createLinkId(args[1]));
        Link endLink = network.getLinks().get(Id.createLinkId(args[2]));

        if (startLink == null || endLink == null) {
            throw new IllegalArgumentException("Start or end link not found.");
        }

        Node startNode = startLink.getToNode();
        Node endNode = endLink.getFromNode();

        Queue<Node> queue = new ArrayDeque<>();
        Set<Id<Node>> visited = new HashSet<>();

        queue.add(startNode);
        visited.add(startNode.getId());

        while (!queue.isEmpty()) {
            Node currentNode = queue.poll();

            for (Link link : currentNode.getOutLinks().values()) {
                if (!link.getAllowedModes().contains(TransportMode.bike)) {
                    continue;
                }

                Node nextNode = link.getToNode();

                if (visited.add(nextNode.getId())) {
                    queue.add(nextNode);
                }
            }
        }

        boolean connected = visited.contains(endNode.getId());

        System.out.println("Start node: " + startNode.getId());
        System.out.println("End node: " + endNode.getId());
        System.out.println("Directed bike route exists: " + connected);
    }
}