/*
 *  felix_exc.h
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

#ifndef FELIX_EXC_H
#define FELIX_EXC_H

// Includes from nestkernel:
#include "archiving_node.h"
#include "connection.h"
#include "event.h"
#include "nest_types.h"
#include "random_generators.h"
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

Excitatory neuron model for Felix networks

Description
+++++++++++

``felix_exc`` low-pass filters incoming currents and emits a spike-like binary
output :math:`\phi(t)` on each time step as well as a low-pass trace of its own output
`\omega_E(t)` for use in synaptic plasticity [1]_. 

The membrane potential evolves according to 

.. math::

   \tau_{\text{m}} \frac{dV_\text{m}(t)}{dt} = -V_{\text{m}}(t) + k_1 \left(I_{\text{in}}(t) + k_2\eta(t)\right)

where the total input is given by

.. math::

   I_{\text{in}}(t) = I_{\text{syn,exc}}(t) + I_{\text{syn,inh}}(t) + I_{\text{syn,G}}(t) + I_E

with synaptic currents respectively from other excitatory neurons (via ``aes_synapse``), inhibitory neurons, 
and the area-global inhibitory neuron, all weighted with the corresponding synaptic weights, and a constant
input current :math:`I_E`. Furthemore, :math:`eta(t)` is a noise current with a value drawn at random uniformly 
from :math:`[-0.5, 0.5)` on each time step.

The output signal of the neuron is given by

.. math::

   \phi(t) = \begin{cases}
                  1 & \text{if}\quad V_m(t) > \theta + \alpha\omega(t) \\
                  0 & \text{otherwise}\;.
             \end{cases}

The adaptive threshold is governed by

.. math::

   \tau_{\text{Adapt}} \frac{\omega(t)}{dt} = -\omega(t) + \phi(t)

and for use by the ABS plasticity rule, the trace

   \tau_{\text{Favg}} \frac{\omega_E(t)}{dt} = -\omega_E(t) + \phi(t)

is computed.

.. note::

   This neuron, as well as ``felix_inh``, sends a spike on every time step. The spike-time offset :math:`O`
   sent with the spike contains the following information:

   * :math:`\phi=1` if :math:`O > 0`, otherwise `\phi=1`
   * :math:`\omega_E=|O|`

   This neuron receives input from other excitatory neurons via ``receptor_type`` 1 and 
   from inhibitory neurons via ``receptor_type`` 2.

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
 ``k_2``         ??       :math:`k_2`                    Scaling factor for noise current
 ``tau_e``       ms       :math:`\tau{\text{Favg}}`      Time constant for plasticity trace
 ``tau_adapt``   ms       :math:`\tau{\text{Adapt}}`     Time constant for adaptive threshold
 ``alpha``       ??       :math:`\alpha`                 Scaling factor for adaptive threshold
 ``alpha_e``     ??       :math:`\alpha_e`                 Scaling factor for estimate firing rate
 ``thresh``      ??       :math:`\theta`                 Fixed threshold component
 ``magic``       ??       ??			                 Regulates spiking behaviour
 Jexcitatory     ??       ??                             Scaling Factor for Excitatory Neurons
 
 Jinhibitory     ??       ??                              Regulation factor for Inhibition (local and global)
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

felix_inh, abs_synapse

EndUserDocs */
// clang-format on
class felix_exc : public nest::StructuralPlasticityNode
{
public:
  /**
   * The constructor is only used to create the model prototype in the model
   * manager.
   */
  felix_exc();

  /**
   * The copy constructor is used to create model copies and instances of the
   * model.
   * @node The copy constructor needs to initialize the parameters and the
   * state.
   *       Initialization of buffers and interal variables is deferred to
   *       @c init_buffers_() and @c pre_run_hook().
   */
  felix_exc( const felix_exc& );

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
  friend class nest::RecordablesMap< felix_exc >;
  friend class nest::UniversalDataLogger< felix_exc >;

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
    double tau_adapt; //!< time constant for om integration
    double tau_e;   //!< time constant for om_e integration
    double I_e;     //!< Intrinsic DC current, in nA.
    double k_1;
    double k_2;
    double alpha;
    double alpha_e;
    double thresh;
	double magic;
    double Jexcitatory;
    //double Jinhibitory;

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
    double om;
    double om_e;
    double I_tot;
    double phi;
    double I_exc;
    double I_inh;
    double I_noise;

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
    Buffers_( felix_exc& );
    Buffers_( const Buffers_&, felix_exc& );

    nest::RingBuffer exc_spikes;   //!< 0/1 spikes from excitatory neurons
    nest::RingBuffer inh_spikes;   //!< Graded currents from inhibitory neurons

    //! Logger for all analog data
    nest::UniversalDataLogger< felix_exc > logger_;
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
    double P_om_adapt;
    double P_om_adapt_phi;
    double P_om_e;
    double P_om_e_phi;

    nest::uniform_real_distribution uni_dist;
  };

  /**
   * @defgroup Access functions for UniversalDataLogger.
   * @{
   */
  //! Read out the real membrane potential
 public:
  double
  get_V_m() const
  {
    return S_.V_m;
  }

 private:
  double
  get_om_() const
  {
    return S_.om;
  }

  double
  get_om_e_() const
  {
    return S_.om_e;
  }

  double
  get_I_tot_() const
  {
    return S_.I_tot;
  }

  double
    get_phi_() const
  {
    return S_.phi;
  }
    
 double
     get_I_exc_() const
   {
     return S_.I_exc;
   }
    
  double
      get_I_inh_() const
    {
      return S_.I_inh;
    }
    
 double
    get_I_noise_() const
      
    {
      return S_.I_noise;
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
  static nest::RecordablesMap< felix_exc > recordablesMap_;

  /** @} */
};

inline size_t
felixmodule::felix_exc::send_test_event( nest::Node& target, size_t receptor_type, nest::synindex, bool )
{
  // You should usually not change the code in this function.
  // It confirms that the target of connection @c c accepts @c SpikeEvent on
  // the given @c receptor_type.
  nest::SpikeEvent e;
  e.set_sender( *this );
  return target.handles_test_event( e, receptor_type );
}

inline size_t
felixmodule::felix_exc::handles_test_event( nest::SpikeEvent&, size_t receptor_type )
{
  // You should usually not change the code in this function.
  // It confirms to the connection management system that we are able
  // to handle @c SpikeEvent on port 0. You need to extend the function
  // if you want to differentiate between input ports.
    if ( receptor_type < 1 or receptor_type > 3 )

      {
        throw nest::UnknownReceptorType( receptor_type, get_name() );
      }
  return receptor_type;
}

inline size_t
felixmodule::felix_exc::handles_test_event( nest::DataLoggingRequest& dlr, size_t receptor_type )
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
felix_exc::get_status( DictionaryDatum& d ) const
{
  // get our own parameter and state data
  P_.get( d );
  S_.get( d );

  // get information managed by parent class
  StructuralPlasticityNode::get_status( d );

  ( *d )[ nest::names::recordables ] = recordablesMap_.get_list();
}

inline void
felix_exc::set_status( const DictionaryDatum& d )
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
  bool felix_exc::is_off_grid() const
{
  return true;
}

 
} // namespace

#endif /* #ifndef FELIX_EXC_H */
