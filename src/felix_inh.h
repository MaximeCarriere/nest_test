/*
 *  felix_inh.h
 *
 *  This file is part of NEST.
 *
 *  Copyright (C) 2004 The NEST Initiative
 *
 *  NEST is free software: you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation, either version 2 of the License, or
 *  (at your option) any later version.
 *
 *  NEST is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with NEST.  If not, see <http://www.gnu.org/licenses/>.
 *
 */

#ifndef FELIX_INH_H
#define FELIX_INH_H

// Includes from nestkernel:
#include "archiving_node.h"
#include "connection.h"
#include "event.h"
#include "nest_types.h"
#include "ring_buffer.h"
#include "universal_data_logger.h"

// Includes from sli:
#include "dictdatum.h"

namespace felixmodule
{


// clang-format off
/* BeginUserDocs: neuron

Short description
+++++++++++++++++

Inhibitory neuron model for Felix networks

Description
+++++++++++

``felix_inh`` low-pass filters input received from ``felix_exc`` neurons
and returns its rectified membrane potential as stepwise continuous output [1]_.

In particular, the membrane potential evolves according to 

.. math::

   \tau_{\text{m}} \frac{dV_\text{m}(t)}{dt} = -V_{\text{m}}(t) + k_1 I_{\text{syn}}(t) 

where the synaptic input current :math:`I_{\text{syn}}(t)` is the total synaptic input
current at a given time step

.. math::

   I_{text{syn}}(t) = \sum_j w_j \phi_j(t - \Delta)

where the sum is over all presynaptic neurons (usually just one for Felix models),
:math:`w_j` the synaptic weight of the connection (usually :math:`w_j=1`), :math:`\phi_j`
the output function of the presynaptic neuron and :math:`\Delta` the delay, 
usually a single time step.

The output of the neuron is given by

.. math::

   \max(0, V_{\text{m}}) \;.

.. note::

   Input values :math:`\phi_j` are received as spike-time offset values and output
   values returned as spike-offset values by spikes emitted on every time step.

.. note::

   This model can also be used to implement the area-global inihibition, but then
   with a non-default time constant.

.. note::

   NEST uses exact integration [2]_ to integrate subthreshold membrane
   dynamics with maximum precision.


Parameters
++++++++++

The following parameters can be set in the status dictionary.

=============== ======== =============================== ========================================================================
**Parameter**   **Unit** **Math equivalent**             **Description**
=============== ======== =============================== ========================================================================
 ``V_m``         mV       :math:`V_{\text{m}}`           Membrane potential
 ``tau_m``       ms       :math:`\tau_{\text{m}}`        Membrane time constant
 ``k_1``         ??       :math:`k_1`                    Scaling factor for input current
=============== ======== =============================== ========================================================================

References
++++++++++

.. [1] Tomasello R, Garagnani M, Wennekers T and Pulvermüller F (2018).
       A Neurobiologically Constrained Cortex Model of Semantic Grounding
       With Spiking Neurons and Brain-Like Connectivity.
       Front. Comput. Neurosci. 12:88. 
       DOI: https::10.3389/fncom.2018.00088
.. [2] Rotter S,  Diesmann M (1999). Exact simulation of
       time-invariant linear systems with applications to neuronal
       modeling. Biologial Cybernetics 81:381-402.
       DOI: https://doi.org/10.1007/s004220050570

See also
++++++++

felix_exc

EndUserDocs */
// clang-format on

class felix_inh : public nest::StructuralPlasticityNode
{
public:
  /**
   * The constructor is only used to create the model prototype in the model
   * manager.
   */
  felix_inh();

  /**
   * The copy constructor is used to create model copies and instances of the
   * model.
   * @node The copy constructor needs to initialize the parameters and the
   * state.
   *       Initialization of buffers and interal variables is deferred to
   *       @c init_buffers_() and @c pre_run_hook().
   */
  felix_inh( const felix_inh& );

  /**
   * Import sets of overloaded virtual functions.
   * This is necessary to ensure proper overload and overriding resolution.
   * @see http://www.gotw.ca/gotw/005.htm.
   */
  using nest::Node::handle;
  using nest::Node::handles_test_event;

  /**
   * Used to validate that we can send SpikeEvent to desired target:port.
   */
  size_t send_test_event( nest::Node&, size_t, nest::synindex, bool ) override;

  /**
   * @defgroup felixmodule_handle Functions handling incoming events.
   * We tell nest that we can handle incoming events of various types by
   * defining @c handle() and @c connect_sender() for the given event.
   * @{
   */
  void handle( nest::SpikeEvent& ) override;         //! accept spikes
  void handle( nest::DataLoggingRequest& ) override; //! allow recording with multimeter

  size_t handles_test_event( nest::SpikeEvent&, size_t ) override;
  size_t handles_test_event( nest::DataLoggingRequest&, size_t ) override;
  /** @} */

  void get_status( DictionaryDatum& ) const override;
  void set_status( const DictionaryDatum& ) override;

  bool is_off_grid() const override;

private:
  //! Reset internal buffers of neuron.
  void init_buffers_() override;

  //! Initialize auxiliary quantities, leave parameters and state untouched.
  void pre_run_hook() override;

  //! Take neuron through given time interval
  void update( nest::Time const&, const long, const long ) override;

  // The next two classes need to be friends to access the State_ class/member
  friend class nest::RecordablesMap< felix_inh >;
  friend class nest::UniversalDataLogger< felix_inh >;

  /**
   * Free parameters of the neuron.
   *
   * These are the parameters that can be set by the user through @c SetStatus.
   * They are initialized from the model prototype when the node is created.
   * Parameters do not change during calls to @c update().
   *
   * @note Parameters_ need neither copy constructor nor @c operator=(), since
   *       all its members are copied properly by the default copy constructor
   *       and assignment operator. Important:
   *       - If Parameters_ contained @c Time members, you need to define the
   *         assignment operator to recalibrate all members of type @c Time .
   *         You may also want to define the assignment operator.
   *       - If Parameters_ contained members that cannot copy themselves, such
   *         as C-style arrays, you need to define the copy constructor and
   *         assignment operator to copy those members.
   */
  struct Parameters_
  {
    double tau_m;   //!< Membrane time constant, in ms.
    double k_1;

    //! Initialize parameters to their default values.
    Parameters_();

    //! Store parameter values in dictionary.
    void get( DictionaryDatum& ) const;

    //! Set parameter values from dictionary.
    void set( const DictionaryDatum& );
  };

  /**
   * Dynamic state of the neuron.
   *
   * These are the state variables that are advanced in time by calls to
   * @c update(). In many models, some or all of them can be set by the user
   * through @c SetStatus. The state variables are initialized from the model
   * prototype when the node is created.
   *
   * @note State_ need neither copy constructor nor @c operator=(), since
   *       all its members are copied properly by the default copy constructor
   *       and assignment operator. Important:
   *       - If State_ contained @c Time members, you need to define the
   *         assignment operator to recalibrate all members of type @c Time .
   *         You may also want to define the assignment operator.
   *       - If State_ contained members that cannot copy themselves, such
   *         as C-style arrays, you need to define the copy constructor and
   *         assignment operator to copy those members.
   */
  struct State_
  {
    double V_m;      //!< Membrane potential, in mV.
    double I_tot;

    /**
     * Construct new default State_ instance based on values in Parameters_.
     * This c'tor is called by the no-argument c'tor of the neuron model. It
     * takes a reference to the parameters instance of the model, so that the
     * state can be initialized in accordance with parameters, e.g.,
     * initializing the membrane potential with the resting potential.
     */
    State_( const Parameters_& );

    /** Store state values in dictionary. */
    void get( DictionaryDatum& ) const;

    /**
     * Set membrane potential from dictionary.
     * @note Receives Parameters_ so it can test that the new membrane potential
     *       is below threshold.
     */
    void set( const DictionaryDatum&, const Parameters_& );
  };

  /**
   * Buffers of the neuron.
   * Ususally buffers for incoming spikes and data logged for analog recorders.
   * Buffers must be initialized by @c init_buffers_(), which is called before
   * @c pre_run_hook() on the first call to @c Simulate after the start of NEST,
   * ResetKernel.
   * @node Buffers_ needs neither constructor, copy constructor or assignment
   *       operator, since it is initialized by @c init_nodes_(). If Buffers_
   *       has members that cannot destroy themselves, Buffers_ will need a
   *       destructor.
   */
  struct Buffers_
  {
    Buffers_( felix_inh& );
    Buffers_( const Buffers_&, felix_inh& );

    nest::RingBuffer exc_spikes;   //!< 0/1 spikes from excitatory neurons

    //! Logger for all analog data
    nest::UniversalDataLogger< felix_inh > logger_;
  };

  /**
   * Internal variables of the neuron.
   * These variables must be initialized by @c pre_run_hook, which is called before
   * the first call to @c update() upon each call to @c Simulate.
   * @node Variables_ needs neither constructor, copy constructor or assignment
   *       operator, since it is initialized by @c pre_run_hook(). If Variables_
   *       has members that cannot destroy themselves, Variables_ will need a
   *       destructor.
   */
  struct Variables_
  {
    double P_V;
    double P_V_input;
  };

  /**
   * @defgroup Access functions for UniversalDataLogger.
   * @{
   */
  //! Read out the real membrane potential
  double
  get_V_m_() const
  {
    return S_.V_m;
  }

  double
  get_I_tot_() const
  {
    return S_.I_tot;
  }

/** @} */

  /**
   * @defgroup pif_members Member variables of neuron model.
   * Each model neuron should have precisely the following four data members,
   * which are one instance each of the parameters, state, buffers and variables
   * structures. Experience indicates that the state and variables member should
   * be next to each other to achieve good efficiency (caching).
   * @note Devices require one additional data member, an instance of the @c
   *       Device child class they belong to.
   * @{
   */
  Parameters_ P_; //!< Free parameters.
  State_ S_;      //!< Dynamic state.
  Variables_ V_;  //!< Internal Variables
  Buffers_ B_;    //!< Buffers.

  //! Mapping of recordables names to access functions
  static nest::RecordablesMap< felix_inh > recordablesMap_;

  /** @} */
};

inline size_t
felixmodule::felix_inh::send_test_event( nest::Node& target, size_t receptor_type, nest::synindex, bool )
{
  // You should usually not change the code in this function.
  // It confirms that the target of connection @c c accepts @c SpikeEvent on
  // the given @c receptor_type.
  nest::SpikeEvent e;
  e.set_sender( *this );
  return target.handles_test_event( e, receptor_type );
}

inline size_t
felixmodule::felix_inh::handles_test_event( nest::SpikeEvent&, size_t receptor_type )
{
  // You should usually not change the code in this function.
  // It confirms to the connection management system that we are able
  // to handle @c SpikeEvent on port 0. You need to extend the function
  // if you want to differentiate between input ports.
  if ( receptor_type != 0 )
  {
    throw nest::UnknownReceptorType( receptor_type, get_name() );
  }
  return receptor_type;
}

inline size_t
felixmodule::felix_inh::handles_test_event( nest::DataLoggingRequest& dlr, size_t receptor_type )
{
  // You should usually not change the code in this function.
  // It confirms to the connection management system that we are able
  // to handle @c DataLoggingRequest on port 0.
  // The function also tells the built-in UniversalDataLogger that this node
  // is recorded from and that it thus needs to collect data during simulation.
  if ( receptor_type != 0 )
  {
    throw nest::UnknownReceptorType( receptor_type, get_name() );
  }

  return B_.logger_.connect_logging_device( dlr, recordablesMap_ );
}

inline void
felix_inh::get_status( DictionaryDatum& d ) const
{
  // get our own parameter and state data
  P_.get( d );
  S_.get( d );

  // get information managed by parent class
  StructuralPlasticityNode::get_status( d );

  ( *d )[ nest::names::recordables ] = recordablesMap_.get_list();
}

inline void
felix_inh::set_status( const DictionaryDatum& d )
{
  Parameters_ ptmp = P_; // temporary copy in case of errors
  ptmp.set( d );         // throws if BadProperty
  State_ stmp = S_;      // temporary copy in case of errors
  stmp.set( d, ptmp );   // throws if BadProperty

  // We now know that (ptmp, stmp) are consistent. We do not
  // write them back to (P_, S_) before we are also sure that
  // the properties to be set in the parent class are internally
  // consistent.
  StructuralPlasticityNode::set_status( d );

  // if we get here, temporaries contain consistent set of properties
  P_ = ptmp;
  S_ = stmp;
}

inline
  bool felix_inh::is_off_grid() const
{
  return true;
}

 
} // namespace

#endif /* #ifndef FELIX_INH_H */
