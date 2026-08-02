#ifndef SMALL_CELL_ENERGY_MODEL_H
#define SMALL_CELL_ENERGY_MODEL_H

#include "ns3/core-module.h"
#include "ns3/energy-module.h"
#include "ns3/object.h"
#include "ns3/log.h"
#include "ns3/nstime.h"
#include "ns3/energy-source.h"
#include "ns3/device-energy-model.h"
#include "ns3/simulator.h"
#include <map>
#include <string>
#include <fstream>  // For CSV writing

namespace ns3 {

class SmallCellEnergyModel : public DeviceEnergyModel
{
public:
  enum SmallCellState
  {
    ACTIVE = 0,
    SM1 = 1,
    SM2 = 2,
    SM3 = 3
  };

  static TypeId GetTypeId(void)
  {
    static TypeId tid = TypeId("ns3::SmallCellEnergyModel")
                          .SetParent<DeviceEnergyModel>()
                          .AddConstructor<SmallCellEnergyModel>()
                          .AddAttribute("InitialEnergy", "Initial energy in joules",
                                        DoubleValue(1000.0),
                                        MakeDoubleAccessor(&SmallCellEnergyModel::m_remainingEnergy),
                                        MakeDoubleChecker<double>())
                          .AddAttribute("ActiveTxPower", "Transmission power in ACTIVE state (W)",
                                        DoubleValue(33.0),
                                        MakeDoubleAccessor(&SmallCellEnergyModel::m_activeTxPower),
                                        MakeDoubleChecker<double>())
                          .AddTraceSource("TotalPower", "Total power consumption",
                                        MakeTraceSourceAccessor(&SmallCellEnergyModel::m_totalPower),
                                        "ns3::TracedValueCallback::Double")
                          .AddTraceSource("TxPower", "Transmission power",
                                        MakeTraceSourceAccessor(&SmallCellEnergyModel::m_txPower),
                                        "ns3::TracedValueCallback::Double")
                          .AddTraceSource("EnergyConsumed", "Total energy consumed",
                                        MakeTraceSourceAccessor(&SmallCellEnergyModel::m_energyConsumedTrace),
                                        "ns3::TracedValueCallback::Double");
    return tid;
  }

  SmallCellEnergyModel()
    : m_currentState(ACTIVE),
      m_remainingEnergy(1000.0),
      m_energyConsumed(0.0),
      m_activeTxPower(33.0),
      m_transitioning(false),
      m_transitionEndTime(Seconds(0))
  {
    double activePower = 20.0; // Reference power for the ACTIVE state.

    // --- Total Power Consumption (Percentage of Active) ---
    m_totalPowerMap[ACTIVE] = activePower;    
    m_totalPowerMap[SM1]    = activePower * 0.69; // 69% of ACTIVE power.
    m_totalPowerMap[SM2]    = activePower * 0.50; // 50% of ACTIVE power.
    m_totalPowerMap[SM3]    = activePower * 0.29; // 29% of ACTIVE power.
    
    // --- Transmission Power Map ---
    m_txPowerMap[ACTIVE] = m_activeTxPower;
    m_txPowerMap[SM1]    = 0;
    m_txPowerMap[SM2]    = 0;
    m_txPowerMap[SM3]    = 0;

    // --- (De)activation Durations ---
    m_activationDelay[SM1] = MilliSeconds(0.5);
    m_activationDelay[SM2] = MilliSeconds(5.0);
    m_activationDelay[SM3] = Seconds(0.5);
    
    m_totalPower = m_totalPowerMap[ACTIVE];
    m_txPower = m_txPowerMap[ACTIVE];
    m_energyConsumedTrace = 0.0;
    m_activeEnteredTime = Simulator::Now();
    m_lastStateChangeTime = Simulator::Now();

    for (int i = ACTIVE; i <= SM3; ++i) {
        m_stateTimeAccumulated[(SmallCellState)i] = 0.0;
    }
  }

  void SetNodeId(uint32_t id) { m_nodeId = id; }
  uint32_t GetNodeId() const { return m_nodeId; }

  void SetState(SmallCellState state)
  {
    if (m_transitioning) {
        std::cout << Simulator::Now().GetSeconds() << "s: [SBS " << m_nodeId
                  << "] Ignoring SetState(" << StateToString(state)
                  << ") because SBS is still transitioning!" << std::endl;
        return;
    }

    const double minActiveDuration = 0.0;
    if (m_currentState == ACTIVE && state != ACTIVE) {
        Time now = Simulator::Now();
        Time timeInActive = now - m_activeEnteredTime;
        if (timeInActive.GetSeconds() < minActiveDuration) {
            double remaining = minActiveDuration - timeInActive.GetSeconds();
            std::cout << now.GetSeconds() << "s: [SBS " << m_nodeId
                      << "] Refusing to leave ACTIVE: Only "
                      << timeInActive.GetSeconds() << "s elapsed, need "
                      << minActiveDuration << "s. Will retry in "
                      << remaining << "s." << std::endl;
            return;
        }
    }

    Time now = Simulator::Now();
    double duration = (now - m_lastStateChangeTime).GetSeconds();
    double energyUsed = m_totalPower * duration;
    m_energyConsumed += energyUsed;
    m_stateTimeAccumulated[m_currentState] += duration;
    m_lastStateChangeTime = now;

    if (m_currentState != state) {
        std::cout << Simulator::Now().GetSeconds() << "s: [SBS " << m_nodeId
                  << "] Changing state from "
                  << StateToString(m_currentState) << " to " << StateToString(state) << std::endl;

        if (state == ACTIVE && m_currentState != ACTIVE) {
            std::cout << Simulator::Now().GetSeconds() << "s: Scheduling activation delay of "
                      << m_activationDelay[m_currentState].GetSeconds()
                      << "s before entering ACTIVE." << std::endl;

            m_transitioning = true;
            m_transitionEndTime = now + m_activationDelay[m_currentState];
            Simulator::Schedule(m_activationDelay[m_currentState],
                                &SmallCellEnergyModel::FinishTransition, this);
        } else {
            m_currentState = state;
            UpdatePowerValues();
            m_transitioning = false;
            m_transitionEndTime = Seconds(0);
            if (state == ACTIVE) {
                m_activeEnteredTime = Simulator::Now();
            }
            // Count state entry
            // m_stateEntryCount[state]++;
        }
    }
  }

  void FinishTransition()
  {
      std::cout << Simulator::Now().GetSeconds() << "s: Finished transition, now ACTIVE." << std::endl;

      // Energy update after activation delay completes
      Time now = Simulator::Now();
      double duration = (now - m_lastStateChangeTime).GetSeconds();
      double energyUsed = m_totalPower * duration;
      m_energyConsumed += energyUsed;
      m_stateTimeAccumulated[m_currentState] += duration;
      m_lastStateChangeTime = now;

      m_currentState = ACTIVE;
      UpdatePowerValues();
      m_transitioning = false;
      m_transitionEndTime = Seconds(0);
      m_activeEnteredTime = Simulator::Now();

      // m_stateEntryCount[ACTIVE]++;
  }

    void FlushFinalState()
  {
    Time now = Simulator::Now();
    double duration = (now - m_lastStateChangeTime).GetSeconds();
    m_energyConsumed += m_totalPower * duration;
    m_stateTimeAccumulated[m_currentState] += duration;
    m_lastStateChangeTime = now;
  }

  SmallCellState GetState() const { return m_currentState; }
  bool IsTransitioning() const { return m_transitioning; }
  Time GetTransitionEndTime() const { return m_transitionEndTime; }
  double GetTotalPowerConsumption() const { return m_totalPower; }
  double GetTransmissionPower() const { return m_txPower; }

  void UpdateEnergyConsumption(Time duration)
  {
    double energyUsed = m_totalPower * duration.GetSeconds();
    m_remainingEnergy -= energyUsed;
    m_energyConsumed += energyUsed;
    m_energyConsumedTrace = m_energyConsumed;
    if (m_remainingEnergy < 0) {
      m_remainingEnergy = 0;
      HandleEnergyDepletion();
    }
  }

  virtual void SetEnergySource(Ptr<EnergySource> source) override { m_energySource = source; }
  virtual double GetTotalEnergyConsumption() const override { return m_energyConsumed; }
  virtual void ChangeState(int newState) override { SetState(static_cast<SmallCellState>(newState)); }
  virtual void HandleEnergyDepletion() override { SetState(SM3); }
  virtual void HandleEnergyRecharged() override { SetState(ACTIVE); }
  virtual void HandleEnergyChanged() override {}

  // Export state counts to CSV for plotting
  void ExportStateTimeToCsv(uint32_t episode)
  {
      std::ofstream file("sbs_state_times.csv", std::ios::app);
      file << episode << "," << m_nodeId;
      for (int i = ACTIVE; i <= SM3; ++i)
          file << "," << m_stateTimeAccumulated[(SmallCellState)i];
      file << std::endl;
      file.close();
  }

  void ResetStateTime()
  {
      for (int i = ACTIVE; i <= SM3; ++i)
          m_stateTimeAccumulated[(SmallCellState)i] = 0.0;
  }

private:
  void UpdatePowerValues()
  {
    m_totalPower = m_totalPowerMap[m_currentState];
    m_txPower = m_txPowerMap[m_currentState];
  }

  static std::string StateToString(SmallCellState state)
  {
    switch (state)
    {
      case ACTIVE: return "ACTIVE";
      case SM1:    return "SM1";
      case SM2:    return "SM2";
      case SM3:    return "SM3";
      default:     return "UNKNOWN";
    }
  }

  SmallCellState m_currentState;
  std::map<SmallCellState, double> m_totalPowerMap;
  std::map<SmallCellState, double> m_txPowerMap;
  std::map<SmallCellState, Time> m_activationDelay;

  double m_remainingEnergy;
  double m_energyConsumed;
  double m_activeTxPower;
  uint32_t m_nodeId;
  TracedValue<double> m_totalPower;
  TracedValue<double> m_txPower;
  TracedValue<double> m_energyConsumedTrace;
  Ptr<EnergySource> m_energySource;

  bool m_transitioning;
  Time m_transitionEndTime;
  Time m_activeEnteredTime;
  Time m_lastStateChangeTime;

  std::map<SmallCellState, double> m_stateTimeAccumulated;
};

} // namespace ns3

#endif // SMALL_CELL_ENERGY_MODEL_H
