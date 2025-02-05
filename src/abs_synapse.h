/*
 *  abs_synapse.h
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

#ifndef ABS_SYNAPSE_H
#define ABS_SYNAPSE_H

// C++ includes:
#include <cmath>

// Includes from nestkernel:
#include "connection.h"

#include "felix_exc.h"

namespace felixmodule
{

/* BeginUserDocs: synapse, spike-timing-dependent plasticity

Short description
+++++++++++++++++

Synapse type for Felix networks

Description
+++++++++++

``abs_synapse`` is synapse model connecting ``felix_exc`` neurons
and implementing LTP and LTD according to Artola, Bröcher and
Singer, as cited by Tomasello et al [1]_. It relies on the 
:math:`\omega_E(t)` trace provided by  ``felix_exc`` neurons.

Parameters
++++++++++

============  ======================================================
 theta_pre    Threshold for pre-synaptic trace
 theta_plus   Upper threshold for post-synaptic membrane potential
 theta_minus  Lower threshold for post-synaptic membrane potential
 Delta        Weight change upon LTP or LTD
 Jmin         Minimum Size for weight
============  ======================================================

The parameters are common to all synapses of the model and must be set using
SetDefaults on the synapse model.

Transmits
+++++++++

SpikeEvent

References
++++++++++

.. [1] Tomasello R, Garagnani M, Wennekers T and Pulvermüller F (2018).
       A Neurobiologically Constrained Cortex Model of Semantic Grounding
       With Spiking Neurons and Brain-Like Connectivity.
       Front. Comput. Neurosci. 12:88. 
       DOI: https::10.3389/fncom.2018.00088

See also
++++++++

felix_exc

EndUserDocs */

/**
 * Class containing the common properties for all synapses of type
 * abs_synapse.
 */
void register_abs_synapse();
  
class ABSCommonProperties : public nest::CommonSynapseProperties
{

public:
  /**
   * Default constructor.
   * Sets all property values to defaults.
   */
  ABSCommonProperties();

  /**
   * Get all properties and put them into a dictionary.
   */
  void get_status( DictionaryDatum& d ) const;

  /**
   * Set properties from the values given in dictionary.
   */
  void set_status( const DictionaryDatum& d, nest::ConnectorModel& cm );

  // data members common to all connections
  double theta_pre;
  double theta_plus;
  double theta_minus;
  double Delta;
};


/**
 * Class representing an STDP connection with homogeneous parameters, i.e.
 * parameters are the same for all synapses.
 */
template < typename targetidentifierT >
class abs_synapse : public nest::Connection< targetidentifierT >
{

public:
  typedef ABSCommonProperties CommonPropertiesType;
  typedef nest::Connection< targetidentifierT > ConnectionBase;

  static constexpr nest::ConnectionModelProperties properties = nest::ConnectionModelProperties::HAS_DELAY
    | nest::ConnectionModelProperties::IS_PRIMARY;
    //| ConnectionModelProperties::SUPPORTS_HPC
    //| ConnectionModelProperties::SUPPORTS_LBL;

  /**
   * Default Constructor.
   * Sets default values for all parameters. Needed by GenericConnectorModel.
   */
  abs_synapse();

  /**
   * Copy constructor from a property object.
   * Needs to be defined properly in order for GenericConnector to work.
   */
  abs_synapse( const abs_synapse& ) = default;
  abs_synapse& operator=( const abs_synapse& ) = default;


  // Explicitly declare all methods inherited from the dependent base
  // ConnectionBase. This avoids explicit name prefixes in all places these
  // functions are used. Since ConnectionBase depends on the template parameter,
  // they are not automatically found in the base class.
  using ConnectionBase::get_delay;
  using ConnectionBase::get_delay_steps;
  using ConnectionBase::get_rport;
  using ConnectionBase::get_target;

  /**
   * Get all properties of this connection and put them into a dictionary.
   */
  void get_status( DictionaryDatum& d ) const;

  /**
   * Set properties of this connection from the values given in dictionary.
   */
  void set_status( const DictionaryDatum& d, nest::ConnectorModel& cm );

  /**
   * Send an event to the receiver of this connection.
   * \param e The event to send
   */
  void send( nest::Event& e, size_t t, const ABSCommonProperties& );

  void
  set_weight( double w )
  {
    weight_ = w;
  }


  class ConnTestDummyNode : public nest::ConnTestDummyNodeBase
  {
  public:
    // Ensure proper overriding of overloaded virtual functions.
    // Return values from functions are ignored.
    using nest::ConnTestDummyNodeBase::handles_test_event;
    size_t
      handles_test_event( nest::SpikeEvent&, size_t ) override
    {
      return nest::invalid_port;
    }
  };

  /*
   * This function calls check_connection on the sender and checks if the
   * receiver accepts the event type and receptor type requested by the sender.
   * Node::check_connection() will either confirm the receiver port by returning
   * true or false if the connection should be ignored.
   * We have to override the base class' implementation, since for STDP
   * connections we have to call register_stdp_connection on the target neuron
   * to inform the Archiver to collect spikes for this connection.
   *
   * \param s The source node
   * \param r The target node
   * \param receptor_type The ID of the requested receptor type
   */
  void
    check_connection( nest::Node& s, nest::Node& t, size_t receptor_type, const CommonPropertiesType& )
  {
    ConnTestDummyNode dummy_target;
    ConnectionBase::check_connection_( dummy_target, s, t, receptor_type );
  }

private:
  // data members of each connection
  double weight_;
};

template < typename targetidentifierT >
constexpr nest::ConnectionModelProperties abs_synapse< targetidentifierT >::properties;

//
// Implementation of class abs_synapse.
//

template < typename targetidentifierT >
abs_synapse< targetidentifierT >::abs_synapse()
  : ConnectionBase() 
  , weight_( 0.1 )
{
}

/**
 * Send an event to the receiver of this connection.
 * \param e The event to send
 * \param p The port under which this connection is stored in the Connector.
 */
template < typename targetidentifierT >
inline void
abs_synapse< targetidentifierT >::send( nest::Event& e, size_t t, const ABSCommonProperties& cp )
//{
//  felix_exc* target = dynamic_cast< felix_exc* >( get_target( t ) );
//  assert( target );
//  assert( get_delay_steps() == 1 );
//  const double om_E = std::abs( e.get_offset() );
//  const double V_m = target->get_V_m();
//
//  if ( om_E > cp.theta_pre )
//  {
//    if ( V_m > cp.theta_plus )
//    {
//     if (weight_<0.225)
//         {
//             weight_ += cp.Delta;
//         }
//    }
//    else if ( V_m > cp.theta_minus )
//    {
//      weight_ -= cp.Delta;
//    }
//  }
//  else if ( V_m > cp.theta_plus )
//  {
//    weight_ -= cp.Delta;
//  }
//
//  e.set_receiver( *target );
//  e.set_weight( weight_ );
//  e.set_delay_steps( get_delay_steps() );
//  e.set_rport( get_rport() );
//  e();
//}



{
  felix_exc* target = dynamic_cast< felix_exc* >( get_target( t ) );
  assert( target );
  assert( get_delay_steps() == 2 );
  //assert( get_delay_steps() == 1 );
  //assert( (get_delay_steps() == 1) || (get_delay_steps() == 2)|| (get_delay_steps() == 3)); // orig. assert( get_delay_steps() == 2 )
  const double om_E = std::abs( e.get_offset() );
  const double V_m = target->get_V_m();

//  if ( V_m > cp.theta_plus )
//  
//      {
//      if ( om_E > cp.theta_pre )
//            {
//             if (weight_<0.225)
//             //if (weight_<0.2)
//                     {
//                         weight_ += cp.Delta;
//                     }
//            }
//        
//      else if ( V_m > cp.theta_minus )
//            {
//                if (weight_>0.00000001)
//                {
//                    weight_ -= cp.Delta;
//                    
//                    if (weight_<0.00000001)
//                        {
//                            weight_ = 0.00000001;
//                        }
//            }
//        }
//      }
//  else if ((om_E > cp.theta_pre) and  ( V_m > cp.theta_minus ) and (weight_>0.00000001))
//  {
//    weight_ -= cp.Delta;
//  }
    if (V_m > cp.theta_plus)  // Check if post-synaptic potential is above LTP threshold
    {
        if (om_E > cp.theta_pre)  // Is there sufficient pre-synaptic activity?
        {
            if (weight_ < 0.225)  // Has the synaptic weight reached its maximum?
            {
                //std::cout << "LTP applied: Weight increased to " << weight_ << std::endl;  // Print LTP event
                weight_ += cp.Delta;  // No: Homosynaptic LTP
                // You might want to update totLTP here if needed.
            }
        }
        else  // "Low" pre-synaptic activity ==> LTD
        {
            if (weight_ > 0.00000001)  // Reached minimum synaptic weight?
            {
                weight_ -= cp.Delta;  // No: "low"-homo or hetero LTD
                // You might want to update totLTD here if needed.

                if (weight_ < 0.00000001)
                {
                    weight_ = 0.00000001;  // Ensure weight does not go below the minimum.
                }
            }
        }
    }
    else if (V_m <= cp.theta_plus)// Check if the conditions are right for LTD*/
    {
        if (om_E > cp.theta_pre && V_m > cp.theta_minus && weight_ > 0.00000001)
        {
            weight_ -= cp.Delta;  // Yes: Homosynaptic LTD
            // You might want to update totLTD here if needed.
            //std::cout << "LTD applied: V_m: " << V_m << ", om_E: " << om_E
            //                  << ", Weight decreased to: " << weight_ << std::endl;
            

            if (weight_ < 0.00000001)
            {
                weight_ = 0.00000001;  // Ensure weight does not go below the minimum.
            }
        }
    }
    

    else {
        weight_ -=0;
    }


  e.set_receiver( *target );
  e.set_weight( weight_ );
  e.set_delay_steps( get_delay_steps() );
  e.set_rport( get_rport() );
  e();
}

template < typename targetidentifierT >
void
abs_synapse< targetidentifierT >::get_status( DictionaryDatum& d ) const
{

  // base class properties, different for individual synapse
  ConnectionBase::get_status( d );
  def< double >( d, nest::names::weight, weight_ );
}

template < typename targetidentifierT >
void
abs_synapse< targetidentifierT >::set_status( const DictionaryDatum& d, nest::ConnectorModel& cm )
{
  // base class properties
  ConnectionBase::set_status( d, cm );
  updateValue< double >( d, nest::names::weight, weight_ );
}

} // of namespace felixmodule

#endif // of #ifndef ABS_SYNAPSE_H
