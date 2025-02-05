/*
 *  felix_exc.cpp
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

#include "felix_exc.h"

// C++ includes:
#include <limits>
#include <cmath>

// Includes from libnestutil:
#include "numerics.h"

// Includes from nestkernel:
#include "exceptions.h"
#include "kernel_manager.h"
#include "universal_data_logger_impl.h"

// Includes from sli:
#include "dict.h"
#include "dictutils.h"
#include "doubledatum.h"
#include "integerdatum.h"
#include "sharedptrdatum.h"

using namespace nest;

/* ----------------------------------------------------------------
 * Recordables map
 * ---------------------------------------------------------------- */

nest::RecordablesMap< felixmodule::felix_exc > felixmodule::felix_exc::recordablesMap_;

namespace nest
{
// Override the create() method with one call to RecordablesMap::insert_()
// for each quantity to be recorded.
template <>
void
RecordablesMap< felixmodule::felix_exc >::create()
{
  // use standard names whereever you can for consistency!
  insert_( names::V_m, &felixmodule::felix_exc::get_V_m );
  insert_( "om", &felixmodule::felix_exc::get_om_ );
  insert_( "om_e", &felixmodule::felix_exc::get_om_e_ );
  insert_( "I_tot", &felixmodule::felix_exc::get_I_tot_ );
  insert_( "phi", &felixmodule::felix_exc::get_phi_ );
  insert_( "I_exc", &felixmodule::felix_exc::get_I_exc_ );
  insert_( "I_inh", &felixmodule::felix_exc::get_I_inh_ );
  insert_( "I_noise", &felixmodule::felix_exc::get_I_noise_ );
  //insert_( "I_pg", &nest::nest::poisson_generator::get );
}
}


/* ----------------------------------------------------------------
 * Default constructors defining default parameters and state
 * ---------------------------------------------------------------- */

felixmodule::felix_exc::Parameters_::Parameters_()
  : tau_m( 2.5 )
  , tau_adapt( 10. )
  , tau_e( 30.0 )
  , I_e( 0 )
  , k_1( 0.01 )
  , k_2( 1 / std::sqrt(24 / nest::Time::Time::get_resolution().get_ms() ) ) // noise
  , alpha( 7.0 )
  , alpha_e (3.0) // alpha for estimate firing rate (important for learning rules)
  , thresh( 0.18 )
  , magic( 872. )
  , Jexcitatory ( 500 )
  //, Jinhibitory ( 20 )
{
}

felixmodule::felix_exc::State_::State_( const Parameters_& p )
  : V_m( 0.0 )
  , om( 0.0 )
  , om_e( 0.0 )
  , I_tot( 0.0 )
  , phi( 0.0 )
  , I_exc( 0.0 )
  , I_inh( 0.0 )
  //, I_pg(0.0)
{
}

/* ----------------------------------------------------------------
 * Parameter and state extractions and manipulation functions
 * ---------------------------------------------------------------- */

void
felixmodule::felix_exc::Parameters_::get( DictionaryDatum& d ) const
{
  ( *d )[ names::tau_m ] = tau_m;
  ( *d )[ "tau_adapt" ] = tau_adapt;
  ( *d )[ "tau_e" ] = tau_e;
  ( *d )[ names::I_e ] = I_e;
  ( *d )[ "k_1" ] = k_1;
  ( *d )[ "k_2" ] = k_2;
  ( *d )[ "alpha" ] = alpha;
  ( *d )[ "alpha_e" ] = alpha_e;
  ( *d )[ "thresh" ] = thresh;
  ( *d )[ "magic" ] = magic;
  ( *d )[ "Jexcitatory" ] = Jexcitatory;
  //( *d )[ "Jinhibitory" ] = Jinhibitory;
    
    
}

void
felixmodule::felix_exc::Parameters_::set( const DictionaryDatum& d )
{
  updateValue< double >( d, names::tau_m, tau_m );
  updateValue< double >( d, "tau_adapt", tau_adapt );
  updateValue< double >( d, "tau_e", tau_e );
  updateValue< double >( d, names::I_e, I_e );
  updateValue< double >( d, "k_1", k_1 );
  updateValue< double >( d, "k_2", k_2 );
  updateValue< double >( d, "alpha", alpha );
  updateValue< double >( d, "alpha_e", alpha_e);
  updateValue< double >( d, "thresh", thresh );
  updateValue< double >( d, "magic", magic );
  updateValue< double >( d, "Jexcitatory", Jexcitatory);
  //updateValue< double >( d, "Jinhibitory", Jinhibitory);
    
  if ( tau_m <= 0 )
  {
    throw nest::BadProperty( "The membrane capacitance must be strictly positive." );
  }
}

void
felixmodule::felix_exc::State_::get( DictionaryDatum& d ) const
{
  ( *d )[ names::V_m ] = V_m;
  ( *d )[ "om" ] = om;
  ( *d )[ "om_e" ] = om_e;
  ( *d )[ "I_tot" ] = I_tot;
  ( *d )[ "phi" ] = phi;
  ( *d )[ "I_exc" ] = I_exc;
  ( *d )[ "I_inh" ] = I_inh;
  ( *d )[ "I_noise" ] = I_noise;
  //( *d )[ "I_pg" ] = I_pg;
}

void
felixmodule::felix_exc::State_::set( const DictionaryDatum& d, const Parameters_& p )
{
  updateValue< double >( d, names::V_m, V_m );
  updateValue< double >( d, "om", om );
  updateValue< double >( d, "om_e", om_e );
}

felixmodule::felix_exc::Buffers_::Buffers_( felix_exc& n )
  : logger_( n )
{
}

felixmodule::felix_exc::Buffers_::Buffers_( const Buffers_&, felix_exc& n )
  : logger_( n )
{
}


/* ----------------------------------------------------------------
 * Default and copy constructor for node
 * ---------------------------------------------------------------- */

felixmodule::felix_exc::felix_exc()
  : StructuralPlasticityNode()
  , P_()
  , S_( P_ )
  , B_( *this )
{
  recordablesMap_.create();
}

felixmodule::felix_exc::felix_exc( const felix_exc& n )
  : StructuralPlasticityNode( n )
  , P_( n.P_ )
  , S_( n.S_ )
  , B_( n.B_, *this )
{
}

/* ----------------------------------------------------------------
 * Node initialization functions
 * ---------------------------------------------------------------- */

void
felixmodule::felix_exc::init_buffers_()
{
  B_.exc_spikes.clear();   // includes resize
  B_.inh_spikes.clear();   // includes resize
  B_.logger_.reset();  // includes resize
}

void
felixmodule::felix_exc::pre_run_hook()
{
  B_.logger_.init();

  const double h = Time::get_resolution().get_ms();
  /* Exact integration */
  V_.P_V = std::exp( -h / P_.tau_m );
  V_.P_V_input = -numerics::expm1( -h / P_.tau_m );
  V_.P_om_adapt = std::exp( -h / P_.tau_adapt );
  V_.P_om_adapt_phi = -numerics::expm1( -h / P_.tau_adapt );
  V_.P_om_e = std::exp( -h / P_.tau_e );
  V_.P_om_e_phi = -numerics::expm1( -h / P_.tau_e );
  
  
  /* Forward Euler
  V_.P_V = 1-h / P_.tau_m;
  V_.P_V_input = h / P_.tau_m ;
  V_.P_om_adapt =1- h / P_.tau_adapt;
  V_.P_om_adapt_phi = h / P_.tau_adapt ;
  V_.P_om_e = 1-h / P_.tau_e ;
  V_.P_om_e_phi = h / P_.tau_e ;
*/
}

/* ----------------------------------------------------------------
 * Update and spike handling functions
 * ---------------------------------------------------------------- */

void
felixmodule::felix_exc::update( Time const& slice_origin, const long from_step, const long to_step )
{
  for ( long lag = from_step; lag < to_step; ++lag )
  {
    // order is important in this loop, since we have to use the old values
    // (those upon entry to the loop) on right hand sides everywhere
      
    S_.I_exc = B_.exc_spikes.get_value( lag );
    S_.I_inh = B_.inh_spikes.get_value( lag );
    
      


    S_.I_noise = 1 * P_.k_2 * ( V_.uni_dist( get_vp_specific_rng( get_thread() ) ) - 0.5 );
    S_.I_tot = P_.k_1 * ( S_.I_exc * P_.Jexcitatory + S_.I_inh  + S_.I_noise + P_.I_e );
    S_.V_m = V_.P_V * S_.V_m + V_.P_V_input * S_.I_tot;


	S_.phi = S_.V_m - P_.magic * S_.om < P_.thresh ? 0 : 1;

    //S_.om = V_.P_om_adapt * S_.om + (S_.phi * P_.alpha);
    //S_.om_e = V_.P_om_e * S_.om_e + (S_.phi * P_.alpha_e);
      
      
    S_.om = V_.P_om_adapt * S_.om + V_.P_om_adapt_phi * S_.phi * P_.alpha;
    //S_.om_e = V_.P_om_e * S_.om_e + V_.P_om_e_phi * S_.phi;// * P_.alpha;
    S_.om_e = V_.P_om_e * S_.om_e + V_.P_om_e_phi * S_.phi;
      
    //S_.om_e = V_.P_om_e * S_.om_e + S_.phi * P_.alpha;


    // send spike, and set spike time in archive.
    set_spiketime( Time::step( slice_origin.get_steps() + lag + 1 ) );
    SpikeEvent se;
    se.set_offset( S_.phi == 1 ? S_.om_e : -S_.om_e ); //if no spike, send negative value - 'fake spike'
    kernel().event_delivery_manager.send( *this, se, lag );

    // log membrane potential
    B_.logger_.record_data( slice_origin.get_steps() + lag );
  }
}

void
felixmodule::felix_exc::handle( SpikeEvent& e )
{
  //assert( e.get_delay_steps() == 2 );
  
  
  const auto rport = e.get_rport();
    if (rport == 1) // or rport == 3 )
  {
    assert( e.get_delay_steps() == 2 );
    //assert( (e.get_delay_steps() == 2) || (e.get_delay_steps() == 1) );
    //assert( (e.get_delay_steps() == 1) || (e.get_delay_steps() == 2) || (e.get_delay_steps() == 3)); // original: 2!!!
    assert( e.get_multiplicity() == 1 );
    if ( e.get_offset() > 0 )
    {
      // phi == 1 in sender
      B_.exc_spikes.add_value(
        e.get_rel_delivery_steps( kernel().simulation_manager.get_slice_origin() ), e.get_weight() );
    }
  }
  else if ( rport == 2 )
  {
    assert( e.get_delay_steps() == 1 );
    assert( e.get_multiplicity() == 1 );
    B_.inh_spikes.add_value(
			    e.get_rel_delivery_steps( kernel().simulation_manager.get_slice_origin() ), e.get_weight() * e.get_offset() );
    // spike from felix_inh
  }
    
  else if ( rport == 3 )
  {
    //assert( e.get_delay_steps() == 1 );
    assert( (e.get_delay_steps() == 1) || (e.get_delay_steps() == 2) );
    B_.exc_spikes.add_value(
        e.get_rel_delivery_steps( kernel().simulation_manager.get_slice_origin() ), e.get_weight() );

  }
  else
  {
    assert( false );
  }
}

// Do not move this function as inline to h-file. It depends on
// universal_data_logger_impl.h being included here.
void
felixmodule::felix_exc::handle( DataLoggingRequest& e )
{
  B_.logger_.handle( e ); // the logger does this for us
}
