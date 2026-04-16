/****************************************************************************
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
*******************************************************************************/

#include "astra-sim/common/Logging.hh"
#include "common/CmdLineParser.hh"
#include "congestion_aware/CongestionAwareNetworkApi.hh"
#include <astra-network-analytical/common/EventQueue.h>
#include <astra-network-analytical/common/NetworkParser.h>
#include <astra-network-analytical/congestion_aware/Helper.h>
#include <astra-network-analytical/congestion_aware/Link.h>
#include <remote_memory_backend/analytical/AnalyticalRemoteMemory.hh>

using namespace AstraSim;
using namespace Analytical;
using namespace AstraSimAnalytical;
using namespace AstraSimAnalyticalCongestionAware;
using namespace NetworkAnalytical;
using namespace NetworkAnalyticalCongestionAware;

int main(int argc, char* argv[]) {
    auto cmd_line_parser = CmdLineParser(argv[0]);
    cmd_line_parser.parse(argc, argv);

    const auto workload_configuration =
        cmd_line_parser.get<std::string>("workload-configuration");
    const auto comm_group_configuration =
        cmd_line_parser.get<std::string>("comm-group-configuration");
    const auto system_configuration =
        cmd_line_parser.get<std::string>("system-configuration");
    const auto remote_memory_configuration =
        cmd_line_parser.get<std::string>("remote-memory-configuration");
    const auto network_configuration =
        cmd_line_parser.get<std::string>("network-configuration");
    const auto logging_configuration =
        cmd_line_parser.get<std::string>("logging-configuration");
    const auto logging_folder =
        cmd_line_parser.get<std::string>("logging-folder");
    const auto num_queues_per_dim =
        cmd_line_parser.get<int>("num-queues-per-dim");
    const auto comm_scale = cmd_line_parser.get<double>("comm-scale");
    const auto injection_scale = cmd_line_parser.get<double>("injection-scale");
    const auto rendezvous_protocol =
        cmd_line_parser.get<bool>("rendezvous-protocol");

    AstraSim::LoggerFactory::init(logging_configuration, logging_folder);

    const auto event_queue = std::make_shared<EventQueue>();
    Topology::set_event_queue(event_queue);

    const auto network_parser = NetworkParser(network_configuration);
    Link::configure_quantization(network_parser.get_quantization_enabled(),
                                 network_parser.get_quantization_ratio(),
                                 network_parser.get_quantization_queue_threshold(),
                                 network_parser.get_quantization_policy(),
                                 network_parser.get_quantization_metric(),
                                 network_parser.get_quantization_threshold(),
                                 network_parser.get_quantization_metadata_mode(),
                                 network_parser.get_quantization_metadata_bytes_per_chunk(),
                                 network_parser.get_quantization_metadata_bytes_per_group(),
                                 network_parser.get_quantization_metadata_group_size_bytes());
    const auto topology = construct_topology(network_parser);

    const auto npus_count = topology->get_npus_count();
    const auto npus_count_per_dim = topology->get_npus_count_per_dim();
    const auto dims_count = topology->get_dims_count();

    CongestionAwareNetworkApi::set_event_queue(event_queue);
    CongestionAwareNetworkApi::set_topology(topology);

    auto network_apis = std::vector<std::unique_ptr<CongestionAwareNetworkApi>>();
    const auto memory_api =
        std::make_unique<AnalyticalRemoteMemory>(remote_memory_configuration);
    auto systems = std::vector<Sys*>();

    auto queues_per_dim = std::vector<int>();
    for (auto i = 0; i < dims_count; i++) {
        queues_per_dim.push_back(num_queues_per_dim);
    }

    for (int i = 0; i < npus_count; i++) {
        auto network_api = std::make_unique<CongestionAwareNetworkApi>(i);
        auto* const system =
            new Sys(i, workload_configuration, comm_group_configuration,
                    system_configuration, memory_api.get(), network_api.get(),
                    npus_count_per_dim, queues_per_dim, injection_scale,
                    comm_scale, rendezvous_protocol);

        network_apis.push_back(std::move(network_api));
        systems.push_back(system);
    }

    for (int i = 0; i < npus_count; i++) {
        systems[i]->workload->fire();
    }

    while (!event_queue->finished()) {
        event_queue->proceed();
    }

    Link::report_quantization_stats();

    for (auto it : systems) {
        delete it;
    }
    systems.clear();

    AstraSim::LoggerFactory::shutdown();
    return 0;
}
