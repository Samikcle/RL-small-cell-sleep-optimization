#include <fstream>
#include "ns3/core-module.h"
#include "ns3/network-module.h"
#include "ns3/internet-module.h"
#include "ns3/lte-module.h"
#include "ns3/mobility-module.h"
#include "ns3/applications-module.h"
#include "ns3/config-store.h"
#include "ns3/point-to-point-module.h"
#include "ns3/netanim-module.h"
#include "smallCellEnergyModel.h"
#include "ns3/basic-energy-source.h"
#include "ns3/energy-source-container.h"
#include "ns3/device-energy-model-container.h"
#include "ns3/flow-monitor-helper.h"
#include "ns3/opengym-module.h"
#include <map>
#include <vector>
#include <unordered_map>
#include <cmath>
#include <functional>
#include <stdexcept>
#include <numeric>


using namespace ns3;


bool g_isBaselineRun = false;  
// Global tracking maps
std::map<uint32_t, uint32_t> sbsToUeCount;         // SBS NodeId -> UE count
std::map<uint64_t, uint32_t> ueToSbsMap;           // IMSI -> SBS NodeId
std::map<uint16_t, uint32_t> cellIdToNodeId;       // CellId -> SBS NodeId
std::unordered_map<uint32_t, Ptr<SmallCellEnergyModel>> sbsEnergyModels;
std::map<std::pair<uint16_t, uint16_t>, uint64_t> rntiToImsiMap;
std::map<uint64_t, ApplicationContainer> ueServerApps; // Map IMSI to server apps
std::map<uint64_t, ApplicationContainer> ueClientApps; // Map IMSI to client apps

// Global map to store UE IMSI and the corresponding SBS NodeId
std::map<uint32_t, std::vector<uint64_t>> sbsToUeMap;
std::map<uint64_t, bool> ueActivityMap;
std::map<Ptr<UdpServer>, Ptr<Node>> udpServerToUeNode;
std::set<uint32_t> macroNodeIds;
NetDeviceContainer* globalEnbDevs;
std::map<uint32_t, double> sbsSinrAverage;
double macroSinrAverage = 0.0;
NetDeviceContainer* globalUeDevs;

NS_LOG_COMPONENT_DEFINE("SmallCellSleep");
NS_OBJECT_ENSURE_REGISTERED(SmallCellEnergyModel);

// === Callback for connection established ===
void NotifyConnectionEstablishedUe(std::string context, uint64_t imsi,
                                   uint16_t cellId, uint16_t rnti)
{
    if (cellIdToNodeId.find(cellId) == cellIdToNodeId.end()) {
        std::cout << Simulator::Now().GetSeconds()
                  << "s: Unknown CellId " << cellId
                  << " during connection for IMSI " << imsi << std::endl;
        return;
    }

    uint32_t sbsNodeId = cellIdToNodeId[cellId];

    // Avoid duplicates
    if (ueToSbsMap.count(imsi)) {
        uint32_t prevSbs = ueToSbsMap[imsi];
        auto& list = sbsToUeMap[prevSbs];
        list.erase(std::remove(list.begin(), list.end(), imsi), list.end());
    }

    // Update maps
    sbsToUeMap[sbsNodeId].push_back(imsi);
    ueToSbsMap[imsi] = sbsNodeId;

    rntiToImsiMap[std::make_pair(cellId, rnti)] = imsi;
    std::cout << Simulator::Now().GetSeconds()
              << "s: UE with IMSI " << imsi
              << " connected to NodeId: " << sbsNodeId
              << " (CellId " << cellId << ", RNTI " << rnti << ")" << std::endl;
}




void PrintSbsPositions(NodeContainer smallCellEnbs)
{
    std::cout << "=== SBS Positions at " << Simulator::Now().GetSeconds() << "s ===" << std::endl;

    for (uint32_t i = 0; i < smallCellEnbs.GetN(); ++i)
    {
        Ptr<Node> sbs = smallCellEnbs.Get(i);
        uint32_t nodeId = sbs->GetId();  // Get the NodeId of the SBS
        Ptr<MobilityModel> sbsMobility = sbs->GetObject<MobilityModel>();

        if (sbsMobility)
        {
            Vector sbsPos = sbsMobility->GetPosition();
            std::cout << "  SBS NodeId " << nodeId 
                      << " Position: (" << sbsPos.x << ", " 
                      << sbsPos.y << ", " << sbsPos.z << ")" << std::endl;
        }
    }

    // Simulator::Schedule(Seconds(1.0), &PrintSbsPositions, smallCellEnbs);
}

void LogUePositions(Ptr<Node> ueNode, uint32_t imsi) {
    Ptr<MobilityModel> mob = ueNode->GetObject<MobilityModel>();
    Vector pos = mob->GetPosition();

    std::ofstream logFile("ue_positions.log", std::ios_base::app);  // Open in append mode
    if (logFile.is_open()) {
        logFile << "Time " << Simulator::Now().GetSeconds() << "s - "
                << "UE[" << imsi << "] Position: ("
                << pos.x << ", " << pos.y << ", " << pos.z << ")" << std::endl;
        logFile.close();
    }
    
    Simulator::Schedule(Seconds(1.0), &LogUePositions, ueNode, imsi);
}


static std::ofstream&
GetDebugLog(const std::string& filename = "sinr-debug.log",
            bool               append    = false)
{
    static std::ofstream logFile;
    static bool          init = false;

    if (!init)
    {
        std::ios_base::openmode mode = std::ios::out |
                                       (append ? std::ios::app
                                               : std::ios::trunc);

        logFile.open(filename, mode);
        if (!logFile.is_open())
            throw std::runtime_error("Cannot open debug log: " + filename);

        init = true;
    }
    return logFile;          // same instance on every call
}


void NotifyHandoverEndOkUe(std::string context, uint64_t imsi,
    uint16_t cellId, uint16_t rnti)
{
    if (cellIdToNodeId.find(cellId) == cellIdToNodeId.end()) {
    NS_LOG_UNCOND("Unknown CellId " << cellId << " during handover for IMSI " << imsi);
    return;
    }

    uint32_t newSbsNodeId = cellIdToNodeId[cellId];

    // Remove from old SBS
    if (ueToSbsMap.count(imsi)) {
    uint32_t oldSbsNodeId = ueToSbsMap[imsi];
    auto& oldList = sbsToUeMap[oldSbsNodeId];
    oldList.erase(std::remove(oldList.begin(), oldList.end(), imsi), oldList.end());
    }

    // Add to new SBS
    sbsToUeMap[newSbsNodeId].push_back(imsi);
    ueToSbsMap[imsi] = newSbsNodeId;

    // std::cout << Simulator::Now().GetSeconds()
    // << "s: UE IMSI " << imsi
    // << " successfully handed over to NodeId " << newSbsNodeId
    // << " (CellId " << cellId << ", RNTI " << rnti << ")" << std::endl;


    Simulator::Schedule(MilliSeconds(10), [=]() {
        rntiToImsiMap[std::make_pair(cellId, rnti)] = imsi;
    });
}



std::string StateToString(SmallCellEnergyModel::SmallCellState state)
{
    switch (state)
    {
        case SmallCellEnergyModel::ACTIVE: return "ACTIVE";
        case SmallCellEnergyModel::SM1:    return "SM1";
        case SmallCellEnergyModel::SM2:    return "SM2";
        case SmallCellEnergyModel::SM3:    return "SM3";
        default:                           return "UNKNOWN";
    }
}


void PrintRadioParameters(NetDeviceContainer enbDevs, NetDeviceContainer ueDevs)
{
    std::cout << "\n==== Effective Radio Parameters ====\n";

    if (enbDevs.GetN() > 0)
    {
        Ptr<LteEnbNetDevice> enbDev = DynamicCast<LteEnbNetDevice>(enbDevs.Get(0));
        Ptr<LteEnbPhy> phy = enbDev->GetPhy();
        std::cout << "eNB DL Transmission Power: " << phy->GetTxPower() << " dBm" << std::endl;
    }

    if (ueDevs.GetN() > 0)
    {
        Ptr<LteUeNetDevice> ueDev = DynamicCast<LteUeNetDevice>(ueDevs.Get(0));
        Ptr<LteUePhy> phy = ueDev->GetPhy();
        std::cout << "UE Noise Figure: " << phy->GetNoiseFigure() << " dB" << std::endl;
    }

    std::cout << "=====================================\n";
}


void ToggleUeApplications(uint64_t imsi, bool activate)
{
    // Get the applications for this UE
    auto serverApp = ueServerApps.find(imsi);
    auto clientApp = ueClientApps.find(imsi);
    
    if (serverApp != ueServerApps.end() && clientApp != ueClientApps.end()) {
        if (activate) {
            serverApp->second.Start(Seconds(0.1));
            clientApp->second.Start(Seconds(0.1));
        } else {
            serverApp->second.Stop(Seconds(0));
            clientApp->second.Stop(Seconds(0));
        }
    }
}

void UpdateUeActivity(uint64_t imsi, double simulationTime) 
{
    Ptr<UniformRandomVariable> rand = CreateObject<UniformRandomVariable>();
    double nextCheck = rand->GetValue(0.1, 0.3); // Randomly check every 0.1 to 0.3 seconds

    
    Simulator::Schedule(Seconds(nextCheck), [imsi, simulationTime, rand]() {
        double now = Simulator::Now().GetSeconds();
        double simTimeHours = (now / simulationTime) * 24.0;
        // std::cout << "SimTime: " << now << "s  (Hour of day: " << simTimeHours << ")" << std::endl;
        double activeProbability;
        if (simTimeHours >= 0 && simTimeHours < 4) activeProbability = 0.05;
        else if (simTimeHours >= 4 && simTimeHours < 6) activeProbability = 0.1;
        else if (simTimeHours >= 6 && simTimeHours < 9) activeProbability = 0.4;
        else if (simTimeHours >= 9 && simTimeHours < 17) activeProbability = 0.8;
        else if (simTimeHours >= 17 && simTimeHours < 22) activeProbability = 0.5;
        else activeProbability = 0.2;

        bool newStatus = (rand->GetValue() < activeProbability);
        ueActivityMap[imsi] = newStatus;

        // Toggle applications based on new status
        ToggleUeApplications(imsi, newStatus);

        // std::cout << "Time=" << now << "s (Hour=" << simTimeHours
        //           << "): UE IMSI=" << imsi
        //           << (newStatus ? " is now ACTIVE" : " is now INACTIVE") << std::endl;

        UpdateUeActivity(imsi, simulationTime);
    });
}


// Utility: dBm to Watt
double dBmToWatt(double dBm) {
    return std::pow(10.0, (dBm - 30.0) / 10.0);
}

// Utility: Watt to dB
double WattTodB(double watt) {
    return 10.0 * std::log10(watt);
}

// Path loss model
double CalculateRxPowerDbm(double txPowerDbm, double distance, double pathLossExp = 3.5) {
    if (distance < 1.0) distance = 1.0;
    return txPowerDbm - 10 * pathLossExp * std::log10(distance);
}



bool IsMacroNode(uint32_t nodeId) {
    extern std::set<uint32_t> macroNodeIds;
    return macroNodeIds.find(nodeId) != macroNodeIds.end();
}


double CalculateSinr(
    Ptr<Node> ueNode,
    Ptr<Node> candidateEnbNode,
    double candidateTxPowerDbm,
    double pathLossExponent,
    NetDeviceContainer& enbDevs,
    bool enableLog = true 
) {
    Ptr<MobilityModel> ueMob = ueNode->GetObject<MobilityModel>();
    Ptr<MobilityModel> candidateEnbMob = candidateEnbNode->GetObject<MobilityModel>();
    double distance = ueMob->GetDistanceFrom(candidateEnbMob);

    double rxSignalDbm  = CalculateRxPowerDbm(candidateTxPowerDbm, distance, pathLossExponent);
    double rxSignalWatt = dBmToWatt(rxSignalDbm);

    double interferenceWatt = 0.0;
    int activeSmallCellCount = 0;
    for (uint32_t i = 0; i < enbDevs.GetN(); ++i) {
        Ptr<LteEnbNetDevice> enbDev = DynamicCast<LteEnbNetDevice>(enbDevs.Get(i));
        if (!enbDev) continue;

        Ptr<Node> enbNode = enbDev->GetNode();
        uint32_t enbNodeId = enbNode->GetId();

        if (enbNodeId == candidateEnbNode->GetId()) continue;
        if (IsMacroNode(enbNodeId)) continue;

        Ptr<MobilityModel> otherEnbMob = enbNode->GetObject<MobilityModel>();
        double otherDist = ueMob->GetDistanceFrom(otherEnbMob);

        Ptr<LteEnbPhy> enbPhy = enbDev->GetPhy();
        double otherTxPowerDbm = enbPhy ? enbPhy->GetTxPower() : 44.77;

        if (otherTxPowerDbm > 0) 
            ++activeSmallCellCount;


        double rxInterfDbm = CalculateRxPowerDbm(otherTxPowerDbm, otherDist, pathLossExponent);
        double rxInterfWatt = dBmToWatt(rxInterfDbm);

        //interferenceWatt += rxInterfWatt;
    }

        // Impose minimum interference if fewer than 2 active small cells
    if (activeSmallCellCount <= 1 && interferenceWatt < 1e-9) {
        interferenceWatt = 1e-9;
        // if (enableLog) {
                   
        //     std::cout << "[INFO] Enforcing interference floor: 1e-9 W due to low active SBS count (" 
        //               << activeSmallCellCount << ")" << std::endl;
        // }
    }

    double noiseWatt = 1.38e-23 * 290 * 20e6;
    double denom = interferenceWatt + noiseWatt;
    if (denom <= 0.0) denom = 1e-12;

    double sinr = rxSignalWatt / denom;
    double sinrDb = 10.0 * std::log10(sinr);
    return sinrDb;
}

void CustomSinrHandover(
    Ptr<LteHelper> lteHelper, 
    NodeContainer& ueNodes, 
    NetDeviceContainer& enbDevs, 
    NetDeviceContainer& ueDevs, 
    double pathLossExponent, 
    double sinrHysteresisDb = 2.0
) {
    NS_ASSERT(lteHelper != nullptr);

    if (enbDevs.GetN() == 0 || ueDevs.GetN() == 0) {
        std::cout << "[WARNING] Empty device containers, skipping SINR-based handover decision" << std::endl;
        return;
    }

    for (uint32_t i = 0; i < ueDevs.GetN(); ++i) {
        // std::cout << "\n[DEBUG] Processing UE " << i << std::endl;

        Ptr<LteUeNetDevice> ueDevice = DynamicCast<LteUeNetDevice>(ueDevs.Get(i));
        if (!ueDevice) continue;

        Ptr<LteUeRrc> ueRrc = ueDevice->GetRrc();
        if (!ueRrc || ueRrc->GetState() != LteUeRrc::CONNECTED_NORMALLY) continue;

        Ptr<Node> ueNode = ueDevice->GetNode();
        uint32_t ueNodeId = ueNode->GetId();

        // Find serving eNB (by NodeId, not CellId)
        Ptr<LteEnbNetDevice> servingEnb = nullptr;
        double servingSinrDb = -std::numeric_limits<double>::infinity();
        uint32_t servingEnbNodeId = 0;

        Ptr<LteEnbNetDevice> bestEnbDev = nullptr;
        double bestSinrDb = -std::numeric_limits<double>::infinity();
        uint32_t bestEnbNodeId = 0;

        Ptr<LteEnbNetDevice> targetEnbDevice = nullptr;
        uint32_t targetEnbNodeId = 0;

        // Find serving eNB
        for (uint32_t j = 0; j < enbDevs.GetN(); ++j) {
            Ptr<LteEnbNetDevice> enbDevice = DynamicCast<LteEnbNetDevice>(enbDevs.Get(j));
            if (!enbDevice) continue;
            Ptr<Node> enbNode = enbDevice->GetNode();
            if (ueRrc->GetCellId() == enbDevice->GetCellId()) { // can add a map for robustness
                servingEnb = enbDevice;
                servingEnbNodeId = enbNode->GetId();
                Ptr<LteEnbPhy> enbPhy = enbDevice->GetPhy();
                double txPowerDbm = enbPhy ? enbPhy->GetTxPower() : 44.77;
                servingSinrDb = CalculateSinr(ueNode, enbNode, txPowerDbm, pathLossExponent, enbDevs, true); 
                bestSinrDb = servingSinrDb;
                bestEnbDev = servingEnb;
                bestEnbNodeId = servingEnbNodeId;
                // std::cout << "[DEBUG] Found serving eNB (NodeId: " << servingEnbNodeId << ", CellId: " << enbDevice->GetCellId()
                //           << ") SINR: " << servingSinrDb << " dB" << std::endl;
                break;
            }
        }
        if (!servingEnb) continue;

        // Check all neighbor eNBs by NodeId
        for (uint32_t j = 0; j < enbDevs.GetN(); ++j) {
            Ptr<LteEnbNetDevice> enbDevice = DynamicCast<LteEnbNetDevice>(enbDevs.Get(j));
            if (!enbDevice) continue;
            Ptr<Node> enbNode = enbDevice->GetNode();
            uint32_t enbNodeId = enbNode->GetId();
            if (enbNodeId == servingEnbNodeId) continue; // skip serving eNB

            Ptr<LteEnbPhy> enbPhy = enbDevice->GetPhy();
            double txPowerDbm = enbPhy ? enbPhy->GetTxPower() : 44.77;
            double neighborSinrDb = CalculateSinr(ueNode, enbNode, txPowerDbm, pathLossExponent, enbDevs, false); 

            // std::cout << "[DEBUG] Neighbor eNB (NodeId: " << enbNodeId << ", CellId: " << enbDevice->GetCellId()
            //           << ") SINR: " << neighborSinrDb << " dB" << std::endl;

            if (neighborSinrDb > bestSinrDb) {
                bestSinrDb = neighborSinrDb;
                bestEnbDev = enbDevice;
                bestEnbNodeId = enbNodeId;
            }
        }

        uint64_t imsi = ueDevice->GetObject<LteUeNetDevice>()->GetImsi();


        if (bestEnbNodeId != servingEnbNodeId &&
            (bestSinrDb - servingSinrDb) > sinrHysteresisDb &&
            ueRrc->GetState() == LteUeRrc::CONNECTED_NORMALLY &&
            ueRrc->GetRnti() != 0)
        {
            // std::cout << " Triggering SINR-based handover from NodeId " << servingEnbNodeId
            //         << " to NodeId " << bestEnbNodeId << "\n" << std::endl;
            lteHelper->HandoverRequest(Seconds(0), ueDevice, servingEnb, bestEnbDev);
        }
    }

    // Schedule the next handover decision
    const double HANDOVER_CHECK_INTERVAL = 0.001;
    Simulator::Schedule(
        Seconds(HANDOVER_CHECK_INTERVAL),
        &CustomSinrHandover,
        lteHelper, ueNodes, enbDevs, ueDevs, pathLossExponent, sinrHysteresisDb
    );
    // std::cout << "\n[INFO] SINR-based handover decision completed. Next check scheduled in "
    //           << HANDOVER_CHECK_INTERVAL << " seconds.\n" << std::endl;
}


void
InitializeSbsTxPower ()
{
  for (const auto &entry : sbsEnergyModels)
    {
      uint32_t                    nodeId = entry.first;
      Ptr<SmallCellEnergyModel>   model  = entry.second;

 
      double txPowerW   = model->GetTransmissionPower ();   

      double txPowerDbm = (txPowerW > 0)
                            ? txPowerW
                            : 0;                     


      Ptr<Node>            sbsNode = NodeList::GetNode (nodeId);
      Ptr<LteEnbNetDevice> enbDev  =
          sbsNode->GetDevice (0)->GetObject<LteEnbNetDevice> ();
      Ptr<LteEnbPhy>       enbPhy  = enbDev->GetPhy ();
      enbPhy->SetTxPower (txPowerDbm);

      std::cout << "0s: [SBS " << nodeId << "] Initial TxPower set to "
                << txPowerDbm << " dBm"  << std::endl;
    }
}


void SampleSbsSinr(NetDeviceContainer& enbDevs, NetDeviceContainer& ueDevs, double pathLossExponent) {
    double tNow = Simulator::Now().GetSeconds();
    // auto& logFile = GetSbsSinrLog();

    for (const auto& [sbsId, ueList] : sbsToUeMap) {
        double sum = 0.0;
        uint32_t count = 0;

        Ptr<Node> sbsNode = NodeList::GetNode(sbsId);
        if (!sbsNode) {
            continue;
        }

        Ptr<LteEnbNetDevice> enbDev = sbsNode->GetDevice(0)->GetObject<LteEnbNetDevice>();
        double txPowerDbm = enbDev->GetPhy()->GetTxPower();

        for (uint64_t imsi : ueList) {
            // Only consider active UEs
            if (!ueActivityMap[imsi]) continue;

            for (uint32_t i = 0; i < ueDevs.GetN(); ++i) {
                Ptr<LteUeNetDevice> ueDev = ueDevs.Get(i)->GetObject<LteUeNetDevice>();
                if (ueDev->GetImsi() != imsi) continue;

                Ptr<Node> ueNode = ueDev->GetNode();
                double sinr = CalculateSinr(ueNode, sbsNode, txPowerDbm, pathLossExponent, *globalEnbDevs, false);

                if (!std::isnan(sinr) && !std::isinf(sinr)) {
                    sum += sinr;
                    count++;
                }
                break;
            }
        }

        double avg = (count > 0) ? sum / count : 0.0;
        sbsSinrAverage[sbsId] = avg;

        // Log to file
        // logFile << tNow << "," << sbsId << "," << avg << "," << count << "\n";
    }
}

void SampleMacroSinr(NetDeviceContainer& ueDevs, uint32_t macroNodeId, double pathLossExponent) {
    double tNow = Simulator::Now().GetSeconds();

    Ptr<Node> macroNode = NodeList::GetNode(macroNodeId);
    if (!macroNode) return;

    Ptr<LteEnbNetDevice> enbDev = macroNode->GetDevice(0)->GetObject<LteEnbNetDevice>();
    double txPowerDbm = enbDev->GetPhy()->GetTxPower();

    double sum = 0.0;
    uint32_t count = 0;

    for (uint32_t i = 0; i < ueDevs.GetN(); ++i) {
        Ptr<LteUeNetDevice> ueDev = ueDevs.Get(i)->GetObject<LteUeNetDevice>();
        Ptr<Node> ueNode = ueDev->GetNode();

        // Only consider UEs actually connected to macro
        uint64_t imsi = ueDev->GetImsi();
        if (ueToSbsMap.count(imsi) && ueToSbsMap[imsi] == macroNodeId) {
            if (!ueActivityMap[imsi]) continue; 
            double sinr = CalculateSinr(ueNode, macroNode, txPowerDbm, pathLossExponent, *globalEnbDevs, false);
            if (!std::isnan(sinr) && !std::isinf(sinr)) {
                sum += sinr;
                count++;
            }
        }
    }

    macroSinrAverage = (count > 0) ? sum / count : 0.0;
    std::cout << "[Macro BS] Avg SINR at " << tNow << "s = " << macroSinrAverage << " dB (from " << count << " UEs)\n";
}


// === START OF RL ===

class LteGymEnv : public OpenGymEnv
{
public:
    LteGymEnv(const std::vector<Ptr<Node>>& sbsNodes,
              const std::map<uint32_t, Ptr<SmallCellEnergyModel>>& energyModels, double simTime);

    Ptr<OpenGymSpace> GetObservationSpace() override;
    Ptr<OpenGymSpace> GetActionSpace() override;
    Ptr<OpenGymDataContainer> GetObservation() override;
    float GetReward() override;
    bool GetGameOver() override;
    std::string GetExtraInfo() override;
    bool ExecuteActions(Ptr<OpenGymDataContainer> action) override;

private:
    std::vector<Ptr<Node>> m_sbsNodes;
    std::map<uint32_t, Ptr<SmallCellEnergyModel>> m_energyModels;
    double m_simulationTime;

    std::map<uint32_t, Time> m_lastUpdateTimes;
    std::map<uint32_t, double> m_lastEnergyConsumptions;
    std::map<uint32_t, uint32_t> m_activeUeCounts;
    std::map<uint32_t, SmallCellEnergyModel::SmallCellState> m_lastStates;
    double m_lastAvgMacroSinr;
    std::map<uint32_t, double> m_lastSbsSinrAverage;
    double m_globalAvgSINR = 0.0;
    uint32_t m_totalUeCount = 30;
    uint32_t m_currentConnectedUeCount = 0;

    void CollectEnvData();
    double ScalePower(double power);
    double ScaleGlobalSinr(double sinrDb);
};

LteGymEnv::LteGymEnv(const std::vector<Ptr<Node>>& sbsNodes,
                     const std::map<uint32_t, Ptr<SmallCellEnergyModel>>& energyModels,
                     double simTime)
    : m_sbsNodes(sbsNodes), m_energyModels(energyModels), m_simulationTime(simTime)
{
    for (const auto& [id, model] : energyModels) {
        m_lastUpdateTimes[id] = Seconds(0);
        m_lastEnergyConsumptions[id] = 0.0;
        m_activeUeCounts[id] = 0;
        m_lastStates[id] = model->GetState(); 
    }
}

Ptr<OpenGymSpace> LteGymEnv::GetObservationSpace()
{
    std::cout << "GetObservationSpace() called" << std::endl;
    uint32_t dim = m_energyModels.size() * 5;  
    std::vector<uint32_t> shape = {dim};
    float low = 0.0;
    float high = 100000.0; 
    return CreateObject<OpenGymBoxSpace>(low, high, shape, "float32");
}

Ptr<OpenGymSpace> LteGymEnv::GetActionSpace()
{
    uint32_t numSbs = 5;  
    std::vector<uint32_t> shape = {numSbs};  

    float low = 0;   // ACTIVE
    float high = 3;  // SM3
    std::string dtype = TypeNameGet<float>();

    return CreateObject<OpenGymBoxSpace>(low, high, shape, dtype);
}


Ptr<OpenGymDataContainer> LteGymEnv::GetObservation()
{
    std::cout << "GetObservation() called" << std::endl;

    CollectEnvData();  

    std::vector<double> obs;
    for (const auto& [sbsId, model] : m_energyModels) {
        obs.push_back((double)m_activeUeCounts[sbsId]);
        obs.push_back((double)model->GetState());
        obs.push_back(m_lastEnergyConsumptions[sbsId]);  // Instantaneous power
        obs.push_back(m_globalAvgSINR);
        obs.push_back(model->IsTransitioning() ? 1.0 : 0.0);
    }

   

    Ptr<OpenGymBoxContainer<double>> container = CreateObject<OpenGymBoxContainer<double>>(std::vector<uint32_t>{(uint32_t)obs.size()});
    container->SetData(obs);
    return container;
}

void LteGymEnv::CollectEnvData()
{
    Time now = Simulator::Now();
    double simTimeHours = (now.GetSeconds() / m_simulationTime) * 24.0;

    // Update SINR sampling first
    SampleSbsSinr(*globalEnbDevs, *globalUeDevs, 3.5);
    SampleMacroSinr(*globalUeDevs, /*macroNodeId=*/3, 3.5);

    std::cout << "[LOG] SimTime: " << now.GetSeconds()
              << "s  (Hour of day: " << simTimeHours << ")" << std::endl;

    uint32_t totalActiveUEs = 0;
    double totalSINRWeightedSum = 0.0;

    // === Process SBS ===
    for (const auto& [sbsId, model] : m_energyModels) 
    {
        uint32_t activeUes = 0;
        std::cout << "  [SBS " << sbsId << "] UEs mapped: ";
        if (sbsToUeMap.find(sbsId) != sbsToUeMap.end()) {
            for (uint64_t imsi : sbsToUeMap[sbsId]) {
                std::cout << imsi << (ueActivityMap[imsi] ? "(active)" : "(inactive)") << " ";
                if (ueActivityMap[imsi]) {
                    activeUes++;
                }
            }
        } else {
            std::cout << "None";
        }
        std::cout << "| Active UEs: " << activeUes << std::endl;

        m_activeUeCounts[sbsId] = activeUes;

        double power = model->GetTotalPowerConsumption();
        m_lastEnergyConsumptions[sbsId] = power;

        double sinrAvg = (sbsSinrAverage.count(sbsId)) ? sbsSinrAverage[sbsId] : 0.0;
        std::cout << "  [SBS " << sbsId << "] Avg SINR this step: " << sinrAvg << " dB" << std::endl;

        totalActiveUEs += activeUes;
        totalSINRWeightedSum += sinrAvg * activeUes;
    }

    // === Process Macro ===
    uint32_t macroActiveUEs = 0;
    for (uint32_t i = 0; i < globalUeDevs->GetN(); ++i) 
    {
        Ptr<LteUeNetDevice> ueDev = globalUeDevs->Get(i)->GetObject<LteUeNetDevice>();
        uint64_t imsi = ueDev->GetImsi();

        if (ueToSbsMap.count(imsi) && ueToSbsMap[imsi] == 3)  // macroNodeId = 3
        {
            if (ueActivityMap[imsi]) {
                macroActiveUEs++;
            }
        }
    }

    std::cout << "  [Macro] Active UEs: " << macroActiveUEs 
              << ", Avg SINR: " << macroSinrAverage << " dB" << std::endl;

    totalActiveUEs += macroActiveUEs;
    m_currentConnectedUeCount = totalActiveUEs;
    totalSINRWeightedSum += macroSinrAverage * macroActiveUEs;

    // === Compute Global Average SINR ===
    if (totalActiveUEs > 0)
        m_globalAvgSINR = totalSINRWeightedSum / totalActiveUEs;
    else
        m_globalAvgSINR = 0.0;

    std::cout << "[Global Average SINR] " << m_globalAvgSINR << " dB" << std::endl;
}

double LteGymEnv::ScalePower(double power) {
    const double powerBest = 20*0.29;   // SM3 (deep sleep)
    const double powerWorse = 20;       // fully ACTIVE

    if (power <= powerBest) return 1.0;
    if (power >= powerWorse) return 0.0;

    return (powerWorse - power) / (powerWorse - powerBest);
}

double LteGymEnv::ScaleGlobalSinr(double sinrDb) {
    if (sinrDb < 0.0)
        return 0.0;
    else if (sinrDb <= 20.0)
        return sinrDb / 20.0;  // Maps [0, 10] dB → [0.0, 1.0]
    else
        return 1.0;
}


float LteGymEnv::GetReward()
{
    std::cout << "GetReward() called" << std::endl;
    uint32_t numSbs = m_energyModels.size();
    std::cout << "number of BS: " << numSbs << std::endl;

    const double w_power = 0.3 / numSbs;   // Increased per-SBS normalized weight
    const double w_sinr = 0.7;             // Balanced SINR weight

    float totalReward = 0.0;
    double totalPower = 0.0;

    for (const auto& [sbsId, model] : m_energyModels)
    {
        double power = model->GetTotalPowerConsumption();
        totalPower += power;

        double scaledPower = ScalePower(power);
        double sbsReward = w_power * scaledPower;
        totalReward += sbsReward;

        m_lastStates[sbsId] = model->GetState(); // Track last state

        std::cout << "[SBS " << sbsId << "] Power=" << power
                  << "W (scaled=" << scaledPower << "), Reward=" << sbsReward << std::endl;
    }

    int activeUeCount = m_currentConnectedUeCount;
    int totalUeCount = m_totalUeCount;
    double scaledGlobalSinr = ScaleGlobalSinr(m_globalAvgSINR);

    // === SINR Reward Component (weighted by user coverage) ===
    double coverageFactor = (totalUeCount > 0)
        ? static_cast<double>(activeUeCount) / totalUeCount
        : 0.0;

    double sinrMultiplier = 0.5 + 0.5 * coverageFactor;  // range: [0.5, 1.0]
    double sinrReward = scaledGlobalSinr * sinrMultiplier;

    totalReward += w_sinr * sinrReward;

    // === Logging ===
    std::cout << "[Reward Breakdown] SINR=" << (w_sinr * sinrReward) << std::endl;
    std::cout << "[Global SINR] " << m_globalAvgSINR << " dB (scaled: " << scaledGlobalSinr << ")\n";
    std::cout << "[UEs] Active: " << activeUeCount << " / " << totalUeCount << std::endl;
    std::cout << "[Final Reward] " << totalReward << std::endl;

    return totalReward;
}

bool LteGymEnv::ExecuteActions(Ptr<OpenGymDataContainer> action)
{
    std::cout << "ExecuteActions() called" << std::endl;
    std::cout << action << std::endl;
    Ptr<OpenGymBoxContainer<float>> box = DynamicCast<OpenGymBoxContainer<float>>(action);

    if (!box) {
        std::cout << "  [ERROR] Box container is NULL!" << std::endl;
        return false;
    }

    std::vector<float> actions = box->GetData();
    std::cout << "  Action vector size: " << actions.size() << std::endl;
    std::cout << "  Actions: ";
    for (auto a : actions) std::cout << a << " ";
    std::cout << std::endl;

    if (actions.size() != m_energyModels.size()) {
        std::cout << "  [ERROR] Actions size mismatch! Actions: " << actions.size() << ", Expected: " << m_energyModels.size() << std::endl;
        return false;
    }

    size_t i = 0;
    for (const auto& [sbsId, model] : m_energyModels) {
        model->SetState((SmallCellEnergyModel::SmallCellState)actions[i]);
        double txPowerW = model->GetTransmissionPower();
        double txPowerDbm = (txPowerW > 0) ? txPowerW : 0;

        Ptr<Node> sbsNode = NodeList::GetNode(sbsId);
        Ptr<LteEnbNetDevice> enbDev = sbsNode->GetDevice(0)->GetObject<LteEnbNetDevice>();
        Ptr<LteEnbPhy> enbPhy = enbDev->GetPhy();
        enbPhy->SetTxPower(txPowerDbm);

        std::cout << Simulator::Now().GetSeconds()
              << "s: [SBS " << sbsId << "] State: " << StateToString(model->GetState())
              << ", Tx Power: " << txPowerDbm << " dBm" << std::endl;

        ++i;
    }
    return true;
}
 
bool LteGymEnv::GetGameOver()
{
    std::cout << "GetGameOver() called" << std::endl;
    
    double now = Simulator::Now().GetSeconds();
    return now >= m_simulationTime;
}


std::string LteGymEnv::GetExtraInfo()
{
    std::cout << "GetExtraInfo() called" << std::endl;

    double totalEnergy = 0.0;
    double totalPower = 0.0;
    for (const auto& [sbsId, model] : m_energyModels)
    {
        totalEnergy += model->GetTotalEnergyConsumption();
        totalPower += model->GetTotalPowerConsumption();
    }


    double globalSinr = m_globalAvgSINR;
    int activeUeCount = m_currentConnectedUeCount;


    std::ostringstream oss;
    oss << "total_energy=" << totalEnergy
        << ";global_sinr=" << globalSinr
        << ";active_ue=" << activeUeCount
        << ";total_power=" << totalPower;

    return oss.str();
}



// === END RL ENV DEFINITION ===

// === STEP CALLBACK FOR RL ===

void StepCallback(Ptr<OpenGymInterface> gymInterface, double stepTime, double simulationTime)
{
    double now = Simulator::Now().GetSeconds();
    // End when simulation time reached
    if (now >= simulationTime) {
        Simulator::Stop();
        return;
    }

    Simulator::Schedule(Seconds(stepTime), &StepCallback, gymInterface, stepTime, simulationTime);
}


void ScheduleNextStateRead(double envStepTime, Ptr<OpenGymInterface> openGym)
{
  Simulator::Schedule (Seconds(envStepTime), &ScheduleNextStateRead, envStepTime, openGym);
  openGym->NotifyCurrentState();
}

// === END OF RL ===

// === Main Simulation ===
int main(int argc, char *argv[])
{
    char* env = std::getenv("NS3_BASELINE");
    if (env && std::string(env) == "1")
    {
        g_isBaselineRun = true;
        // std::cout << "[INFO] Running in BASELINE mode — SBS metrics won't be saved.\n";
    }
    else
    {
        // std::cout << "[INFO] Running in RL training mode — SBS metrics will be saved.\n";
    }

    uint16_t numSmallCells = 5;
    uint16_t numUesPerCell = 6;
    double simulationTime = 10; // seconds 
    double stepTime = 0.01; // seconds
    double pathLossExponent = 3.5; // Path loss exponent for SINR calculation, 
    double sinrHysteresisDb = 2.0; // Minimum SINR improvement needed for handover (dB)
    // double handoverCheckInterval = 0.2; // seconds
    uint32_t openGymPort = 5555;
    uint32_t simSeed = 1;
    uint32_t testArg = 0;
    CommandLine cmd;
    cmd.AddValue("openGymPort", "OpenGym port", openGymPort);
    cmd.AddValue("simSeed", "Simulation seed", simSeed);
    cmd.AddValue ("simTime", "Simulation time in seconds. Default: 10s", simulationTime);
    cmd.AddValue("stepTime", "RL Step time (seconds)", stepTime);
    cmd.Parse(argc, argv);

    std::cout << "Simulation is running.";
    std::cout << "Ns3Env parameters:" << std::endl;
    std::cout << "--simulationTime: " << simulationTime << std::endl;
    std::cout << "--openGymPort: " << openGymPort << std::endl;
    // std::cout << "--envStepTime: " << envStepTime << std::endl;
    std::cout << "--seed: " << simSeed << std::endl;
    std::cout << "--testArg: " << testArg << std::endl;
    std::cout << "--testArg: " << testArg << std::endl;

    RngSeedManager::SetSeed (1);
    RngSeedManager::SetRun (simSeed);
    

    Ptr<LteHelper> lteHelper = CreateObject<LteHelper>();
    Ptr<PointToPointEpcHelper> epcHelper = CreateObject<PointToPointEpcHelper>();
    lteHelper->SetEpcHelper(epcHelper);

    // lteHelper->SetAttribute("PathlossModel", StringValue("ns3::LogDistancePropagationLossModel"));
    // lteHelper->SetHandoverAlgorithmType("ns3::A3RsrpHandoverAlgorithm");
    // lteHelper->SetHandoverAlgorithmAttribute("Hysteresis", DoubleValue(0));
    // lteHelper->SetHandoverAlgorithmAttribute("TimeToTrigger", TimeValue(MilliSeconds(0)));

    // lteHelper->SetHandoverAlgorithmType("ns3::A2A4RsrqHandoverAlgorithm");
    // lteHelper->SetHandoverAlgorithmAttribute("ServingCellThreshold", UintegerValue(28));  // Choose 0–34; lower is more aggressive
    // lteHelper->SetHandoverAlgorithmAttribute("NeighbourCellOffset", UintegerValue(1));    // Lower is more aggressive

    // lteHelper->SetAttribute("FadingModel", StringValue("ns3::TraceFadingLossModel"));

    Config::SetDefault("ns3::LteUePhy::RsrpSinrSamplePeriod", UintegerValue(1));

    // Config::SetDefault("ns3::LogDistancePropagationLossModel::Exponent", DoubleValue(3.2));
    // Config::SetDefault("ns3::TraceFadingLossModel::TraceFilename",
    //                StringValue("src/lte/model/fading-traces/fading_trace_EVA_60kmph.fad"));
    // Config::SetDefault("ns3::TraceFadingLossModel::WindowSize", TimeValue(Seconds(0.5)));
    // Config::SetDefault("ns3::TraceFadingLossModel::RbNum", UintegerValue(100));
    // Config::SetDefault("ns3::LteSpectrumPhy::NoiseFigure", DoubleValue(9.0));
    // Config::SetDefault("ns3::TraceFadingLossModel::SubBands", UintegerValue(1));
    Ptr<Node> pgw = epcHelper->GetPgwNode();

    NodeContainer macroEnb;
    macroEnb.Create(1);

    NodeContainer smallCellEnbs;
    smallCellEnbs.Create(numSmallCells);

    NodeContainer ueNodes;
    ueNodes.Create(numUesPerCell * numSmallCells);

    // === Mobility Setup ===
    MobilityHelper mobility;
    // MobilityHelper ueMobility;
    // Set static position for the first UE
    // Ptr<ListPositionAllocator> staticAlloc = CreateObject<ListPositionAllocator>();
    // staticAlloc->Add(Vector(250.0, 244.0, 0.0));
    // MobilityHelper staticMobility;
    // staticMobility.SetMobilityModel("ns3::ConstantPositionMobilityModel");
    // staticMobility.SetPositionAllocator(staticAlloc);
    // staticMobility.Install(ueNodes.Get(0));  

    // NodeContainer mobileUes;
    // for (uint32_t i = 1; i < ueNodes.GetN(); ++i) {
    //     mobileUes.Add(ueNodes.Get(i));
    // }
    // Create Position Allocator separately
    Ptr<PositionAllocator> uePositionAlloc = CreateObject<RandomRectanglePositionAllocator>();
    uePositionAlloc->SetAttribute("X", StringValue("ns3::UniformRandomVariable[Min=0.0|Max=2500.0]"));
    uePositionAlloc->SetAttribute("Y", StringValue("ns3::UniformRandomVariable[Min=1300.0|Max=1500.0]"));

    // Configure the UE MobilityHelper with RandomWaypointMobilityModel
    MobilityHelper ueMobility;
    ueMobility.SetMobilityModel("ns3::RandomWaypointMobilityModel",
        "Speed", StringValue("ns3::UniformRandomVariable[Min=1.0|Max=50.0]"),
        "Pause", StringValue("ns3::ConstantRandomVariable[Constant=0.5]"),
        "PositionAllocator", PointerValue(uePositionAlloc));
    ueMobility.SetPositionAllocator(uePositionAlloc);
    ueMobility.Install(ueNodes);

    // MobilityHelper ueMobility;
    // ueMobility.SetMobilityModel("ns3::ConstantPositionMobilityModel");
    // ueMobility.SetPositionAllocator(uePositionAlloc); // for non-manual UEs
    // ueMobility.Install(ueNodes);

    // Override position of UE[0] and UE[1]
    for (uint32_t i = 0; i < ueNodes.GetN(); ++i) {
        Ptr<Node> ueNode = ueNodes.Get(i);
        Ptr<MobilityModel> mob = ueNode->GetObject<MobilityModel>();

        // if (i == 0) {
        //     // IMSI = 1 (first UE)
        //     mob->SetPosition(Vector(250.0, 850.0, 0.0));
        // } else if (i == 1) {
        //     // IMSI = 2 (second UE)
        //     mob->SetPosition(Vector(375.0, 800.0, 0.0));
        // }

        // Log IMSI + position
        uint32_t imsi = i + 1;
        // LogUePositions(ueNode, imsi);
    }




    // --- Macro eNB at the center (0, 0) ---
    Ptr<ListPositionAllocator> centerAlloc = CreateObject<ListPositionAllocator>();
    centerAlloc->Add(Vector(0.0, 0.0, 0.0)); 
    MobilityHelper macroMobility;
    macroMobility.SetMobilityModel("ns3::ConstantPositionMobilityModel");
    macroMobility.SetPositionAllocator(centerAlloc);
    macroMobility.Install(macroEnb);

    // --- 5 Small Cells in a circle around the center ---
    Ptr<ListPositionAllocator> sbsCircleAlloc = CreateObject<ListPositionAllocator>();
    double radius = 500.0; // Radius in meters
    double numSbs = 5.0;

    for (int i = 0; i < numSbs; ++i) {
        double angle = 2.0 * M_PI * i / numSbs;
        double x = radius * cos(angle);
        double y = radius * sin(angle);
        sbsCircleAlloc->Add(Vector(x, y, 0.0));
    }

    MobilityHelper sbsMobility;
    sbsMobility.SetMobilityModel("ns3::ConstantPositionMobilityModel");
    sbsMobility.SetPositionAllocator(sbsCircleAlloc);
    sbsMobility.Install(smallCellEnbs);

    // === LTE Device Installation 
    NetDeviceContainer macroEnbLteDevs = lteHelper->InstallEnbDevice(macroEnb);
    NetDeviceContainer smallCellLteDevs = lteHelper->InstallEnbDevice(smallCellEnbs);
    NetDeviceContainer ueLteDevs = lteHelper->InstallUeDevice(ueNodes);

    NetDeviceContainer enbDevs;
    enbDevs.Add(macroEnbLteDevs);
    enbDevs.Add(smallCellLteDevs);
    globalEnbDevs = &enbDevs;

    NodeContainer allEnbs;
    allEnbs.Add(macroEnb);
    allEnbs.Add(smallCellEnbs);
    lteHelper->AddX2Interface(allEnbs); // Add X2 interface for all eNBs

    PrintRadioParameters(smallCellLteDevs, ueLteDevs);
    for (uint32_t i = 0; i < smallCellEnbs.GetN(); ++i)
    {
        Ptr<Node> scNode = smallCellEnbs.Get(i);

        // Create and install energy source
        Ptr<BasicEnergySource> energySource = CreateObject<BasicEnergySource>();
        energySource->SetInitialEnergy(1000.0);
        scNode->AggregateObject(energySource);

        // Create and install energy model
        Ptr<SmallCellEnergyModel> scEnergyModel = CreateObject<SmallCellEnergyModel>();
        scEnergyModel->SetEnergySource(energySource);
        energySource->AppendDeviceEnergyModel(scEnergyModel);


        uint32_t nodeId = scNode->GetId();
        sbsEnergyModels[nodeId] = scEnergyModel;

        scEnergyModel->SetNodeId(nodeId);
        
        // Simulator::Schedule(Seconds(1.0), &UpdateEnergy, scEnergyModel);
    }


    for (uint32_t i = 0; i < macroEnbLteDevs.GetN (); ++i)
    {
        Ptr<LteEnbNetDevice> enbDev = macroEnbLteDevs.Get (i)->GetObject<LteEnbNetDevice> ();
        Ptr<LteEnbPhy>       enbPhy = enbDev->GetPhy ();

        enbPhy->SetTxPower (33.0);   // value is in dBm
    }
    InitializeSbsTxPower ();
    
    for (uint32_t i = 0; i < macroEnb.GetN(); ++i) {
        macroNodeIds.insert(macroEnb.Get(i)->GetId());
    }
    
    // Build CellId -> NodeId map
    // for (uint16_t i = 0; i < smallCellLteDevs.GetN(); ++i)
    // {
    //     Ptr<LteEnbNetDevice> enbDev = smallCellLteDevs.Get(i)->GetObject<LteEnbNetDevice>();
    //     if (enbDev)
    //     {
    //         uint16_t cellId = enbDev->GetCellId();
    //         uint32_t nodeId = smallCellEnbs.Get(i)->GetId();
    //         cellIdToNodeId[cellId] = nodeId;
    //     }
    // }
    // Register ALL eNBs (macro + small cell) in the cellIdToNodeId map
    for (uint16_t i = 0; i < allEnbs.GetN(); ++i)
    {
        Ptr<Node> enbNode = allEnbs.Get(i);
        Ptr<NetDevice> dev = enbNode->GetDevice(0); // Assume LTE device is at index 0
        Ptr<LteEnbNetDevice> enbDev = dev->GetObject<LteEnbNetDevice>();
        if (enbDev)
        {
            uint16_t cellId = enbDev->GetCellId();
            uint32_t nodeId = enbNode->GetId();
            cellIdToNodeId[cellId] = nodeId;
        }
    }



    // for (const auto& pair : cellIdToNodeId)
    // {
    //     std::cout << "Mapped CellId " << pair.first << " to NodeId " << pair.second << std::endl;
    // }

    // === Internet stack ===
    InternetStackHelper internet;
    internet.Install(ueNodes);
    internet.Install(macroEnb);
    internet.Install(smallCellEnbs);
    Ipv4InterfaceContainer ueIpIface = epcHelper->AssignUeIpv4Address(NetDeviceContainer(ueLteDevs));

    // // === Initial attachment (round-robin) ===
    // for (uint16_t i = 0; i < ueNodes.GetN(); ++i)
    // {
    //     Ptr<NetDevice> ueDev = ueLteDevs.Get(i);
    //     uint16_t smallCellIndex = i % numSmallCells;
    //     Ptr<NetDevice> enbDev = smallCellLteDevs.Get(smallCellIndex);
    
    //     lteHelper->Attach(ueDev, enbDev);
    // }

    for (uint16_t i = 0; i < ueNodes.GetN(); ++i)
    {
        Ptr<Node> ueNode = ueNodes.Get(i);
        Ptr<MobilityModel> ueMobility = ueNode->GetObject<MobilityModel>();

        double minDistance = std::numeric_limits<double>::max();
        Ptr<NetDevice> closestEnbDev = nullptr;

        for (uint16_t j = 0; j < smallCellEnbs.GetN(); ++j)
        {
            Ptr<Node> enbNode = smallCellEnbs.Get(j);
            Ptr<MobilityModel> enbMobility = enbNode->GetObject<MobilityModel>();

            double distance = ueMobility->GetDistanceFrom(enbMobility);
            if (distance < minDistance)
            {
                minDistance = distance;
                closestEnbDev = smallCellLteDevs.Get(j);
            }
        }

        lteHelper->Attach(ueLteDevs.Get(i), closestEnbDev);
    }


    for (uint32_t i = 0; i < ueLteDevs.GetN(); ++i)
    {
        Ptr<LteUeNetDevice> ueDevice = ueLteDevs.Get(i)->GetObject<LteUeNetDevice>();
        Ptr<LteUePhy> uePhy = ueDevice->GetPhy();
        
        
        // uePhy->TraceConnectWithoutContext("ReportCurrentCellRsrpSinr", MakeCallback(&ReportTraceCallback));
    }
   

    // === Attach callback for handover + connection tracking ===
    for (uint32_t i = 0; i < ueLteDevs.GetN(); ++i)
    {
        Ptr<NetDevice> dev = ueLteDevs.Get(i);
        Ptr<LteUeNetDevice> ueLte = dev->GetObject<LteUeNetDevice>();
        Ptr<LteUeRrc> rrc = ueLte->GetRrc();
        rrc->TraceConnect("ConnectionEstablished", "", MakeCallback(&NotifyConnectionEstablishedUe));
        rrc->TraceConnect("HandoverEndOk", "", MakeCallback(&NotifyHandoverEndOkUe));

    }

    

    for (uint32_t i = 0; i < ueNodes.GetN(); ++i)
    {
        Ptr<NetDevice> ueDev = ueLteDevs.Get(i);
        Ptr<LteUeNetDevice> ueLteDev = ueDev->GetObject<LteUeNetDevice>();
        uint64_t imsi = ueLteDev->GetImsi();

        ueActivityMap[imsi] = false; // default to inactive
        
        UpdateUeActivity(imsi, simulationTime);
    
        // Log initial position
        // LogUePositions(ueNodes.Get(i), imsi);
    }

    uint16_t port = 5000;
    ApplicationContainer serverApps, clientApps;


    for (uint32_t i = 0; i < ueNodes.GetN(); ++i)
    {
        Ptr<LteUeNetDevice> ueLteDev = ueLteDevs.Get(i)->GetObject<LteUeNetDevice>();
        uint64_t imsi = ueLteDev->GetImsi();

        // UDP Server
        UdpServerHelper udpServer(port);
        ueServerApps[imsi] = udpServer.Install(ueNodes.Get(i));
        ueServerApps[imsi].Start(Seconds(1.0));
        ueServerApps[imsi].Stop(Seconds(simulationTime));



        // UDP Client
        UdpClientHelper udpClient(ueIpIface.GetAddress(i), port);
        udpClient.SetAttribute("MaxPackets", UintegerValue(32000));
        udpClient.SetAttribute("Interval", TimeValue(MilliSeconds(50)));
        udpClient.SetAttribute("PacketSize", UintegerValue(1024));
        ueClientApps[imsi] = udpClient.Install(pgw);
        ueClientApps[imsi].Start(Seconds(1.1)); // Start slightly after server
        ueClientApps[imsi].Stop(Seconds(simulationTime));
    }


    serverApps.Start(Seconds(1.0));
    clientApps.Start(Seconds(1.0));
    serverApps.Stop(Seconds(simulationTime));
    clientApps.Stop(Seconds(simulationTime));



    Simulator::Schedule(
        Seconds(0.001),
        &CustomSinrHandover,
        lteHelper, ueNodes, enbDevs, ueLteDevs, pathLossExponent, sinrHysteresisDb
    );

    FlowMonitorHelper flowmonHelper;
    Ptr<FlowMonitor> monitor = flowmonHelper.InstallAll();  



    globalUeDevs = &ueLteDevs;
    // === Schedule Logging ===
    // Simulator::Schedule(Seconds(1.0), &PrintRemainingEnergy, energyModels);
    
    
    Simulator::Schedule(Seconds(0.0), &PrintSbsPositions, smallCellEnbs);
    Simulator::Schedule(Seconds(0.0), &PrintSbsPositions, macroEnb);
    // Simulator::Schedule(Seconds(0.5), &PrintSbsConnections);
    // Simulator::Schedule(Seconds(1.0), &PrintSbsUeMappings);
    // Simulator::Schedule(Seconds(1.0), &PrintSbsUeCounts);
    // Simulator::Schedule(Seconds(0.5), &UpdateSbsSleepStates);


    // Simulator::Schedule(Seconds(1.0), &PrintUeToSbsDistances, ueNodes, cellIdToNodeId);
    // Simulator::Schedule(Seconds(1.0), &PrintUePositions, ueNodes);


    // === OPENGYM INTERFACE ===
    std::vector<Ptr<Node>> sbsNodes;
    std::map<uint32_t, Ptr<SmallCellEnergyModel>> energyModels;

    for (uint32_t i = 0; i < smallCellEnbs.GetN(); ++i) {
        Ptr<Node> sbsNode = smallCellEnbs.Get(i);
        uint32_t nodeId = sbsNode->GetId();
        Ptr<SmallCellEnergyModel> model = sbsEnergyModels[nodeId];
        sbsNodes.push_back(sbsNode);
        energyModels[nodeId] = model;
    }

    // Create environment and interface
    Ptr<OpenGymInterface> openGym = CreateObject<OpenGymInterface> (openGymPort);
    Ptr<LteGymEnv> lteEnv = CreateObject<LteGymEnv>(sbsNodes, energyModels, simulationTime);

    openGym->SetGetActionSpaceCb(MakeCallback (&LteGymEnv::GetActionSpace, lteEnv) );
    openGym->SetGetObservationSpaceCb( MakeCallback (&LteGymEnv::GetObservationSpace, lteEnv) );
    openGym->SetGetGameOverCb( MakeCallback (&LteGymEnv::GetGameOver, lteEnv) );
    openGym->SetGetObservationCb( MakeCallback (&LteGymEnv::GetObservation, lteEnv) );
    openGym->SetGetRewardCb( MakeCallback (&LteGymEnv::GetReward, lteEnv) );
    openGym->SetGetExtraInfoCb( MakeCallback (&LteGymEnv::GetExtraInfo, lteEnv) );
    openGym->SetExecuteActionsCb( MakeCallback (&LteGymEnv::ExecuteActions, lteEnv) );
    Simulator::Schedule (Seconds(0.0), &ScheduleNextStateRead, stepTime, openGym);
    // Simulator::Schedule(Seconds(0.0), &SampleSbsSinr, std::ref(enbDevs), std::ref(ueLteDevs), pathLossExponent);
    


    // GetRsrpSinrLog();
    GetDebugLog();


    std::cout <<"Simulation start"<< std::endl;
    Simulator::Stop (Seconds (simulationTime));
    Simulator::Run ();
    std::cout << "Simulation stop";

    openGym->NotifySimulationEnd();
    for (const auto& [sbsId, model] : sbsEnergyModels)
    {
        model->FlushFinalState();
    }
    Simulator::Destroy ();

    // Ptr<LteGymEnv> gymEnv = CreateObject<LteGymEnv>(sbsNodes, energyModels, simulationTime);
    // Ptr<OpenGymInterface> gymInterface = CreateObject<OpenGymInterface>(openGymPort);
    // gymInterface->Notify(gymEnv);

    // // === RL STEP-BASED SCHEDULING ===
    // // Schedule the RL step callback (starts at t=0.0, repeats every stepTime)
    // Simulator::Schedule(Seconds(0.0), &StepCallback, gymInterface, stepTime, simulationTime);
    // Simulator::Run();
    // std::cout << "=== Simulation finished at " << Simulator::Now().GetSeconds() << "s ===" << std::endl;
    // gymInterface->NotifySimulationEnd();
    // Simulator::Destroy();

    if (!g_isBaselineRun)
    {
        for (const auto& [sbsId, model] : sbsEnergyModels)
        {
            model->ExportStateTimeToCsv(1);
        }
    }else
    {
        std::cout << "[INFO] Skipping SBS CSV export in baseline mode.\n";
    }
    // monitor->CheckForLostPackets();
    // monitor->SerializeToXmlFile("flowmon-results.xml", true, true); 


    return 0;
}