/*
 *  felix_inh.cpp
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

#include "felix_inh.h"

// C++ includes:
#include <limits>

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

nest::RecordablesMap< felixmodule::felix_inh > felixmodule::felix_inh::recordablesMap_;

namespace nest
{
// Override the create() method with one call to RecordablesMap::insert_()
// for each quantity to be recorded.
template <>
void
RecordablesMap< felixmodule::felix_inh >::create()
{
  // use standard names whereever you can for consistency!
  insert_( names::V_m, &felixmodule::felix_inh::get_V_m_ );
  insert_( "I_tot", &felixmodule::felix_inh::get_I_tot_ );
}
}


/* ----------------------------------------------------------------
 * Default constructors defining default parameters and state
 * ---------------------------------------------------------------- */

felixmodule::felix_inh::Parameters_::Parameters_()
  : tau_m( 5 )
  , k_1( 0.01 )
{
}

felixmodule::felix_inh::State_::State_( const Parameters_& p )
  : V_m( 0.0 )
  , I_tot( 0.0 )
{
}

/* ----------------------------------------------------------------
 * Parameter and state extractions and manipulation functions
 * ---------------------------------------------------------------- */

void
felixmodule::felix_inh::Parameters_::get( DictionaryDatum& d ) const
{
  ( *d )[ names::tau_m ] = tau_m;
  ( *d )[ "k_1" ] = k_1;
}

void
felixmodule::felix_inh::Parameters_::set( const DictionaryDatum& d )
{
  updateValue< double >( d, names::tau_m, tau_m );
  updateValue< double >( d, "k_1", k_1 );
  if ( tau_m <= 0 )
  {
    throw nest::BadProperty( "The membrane time constant must be strictly positive." );
  }
}

void
felixmodule::felix_inh::State_::get( DictionaryDatum& d ) const
{
  ( *d )[ names::V_m ] = V_m;
  ( *d )[ "I_tot" ] = I_tot;
}

void
felixmodule::felix_inh::State_::set( const DictionaryDatum& d, const Parameters_& p )
{
  updateValue< double >( d, names::V_m, V_m );
}

felixmodule::felix_inh::Buffers_::Buffers_( felix_inh& n )
  : logger_( n )
{
}

felixmodule::felix_inh::Buffers_::Buffers_( const Buffers_&, felix_inh& n )
  : logger_( n )
{
}


/* ----------------------------------------------------------------
 * Default and copy constructor for node
 * ---------------------------------------------------------------- */

felixmodule::felix_inh::felix_inh()
  : StructuralPlasticityNode()
  , P_()
  , S_( P_ )
  , B_( *this )
{
  recordablesMap_.create();
}

felixmodule::felix_inh::felix_inh( const felix_inh& n )
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
felixmodule::felix_inh::init_buffers_()
{
  B_.exc_spikes.clear();   // includes resize
  B_.logger_.reset();  // includes resize
}

void
felixmodule::felix_inh::pre_run_hook()
{
  B_.logger_.init();

  const double h = Time::get_resolution().get_ms();
  V_.P_V = std::exp( -h / P_.tau_m );
  //V_.P_V_input = - P_.tau_m * numerics::expm1( -h / P_.tau_m );
  V_.P_V_input = -numerics::expm1( -h / P_.tau_m );
    
    
}

/* ----------------------------------------------------------------
 * Update and spike handling functions
 * ---------------------------------------------------------------- */

void
felixmodule::felix_inh::update( Time const& slice_origin, const long from_step, const long to_step )
{
  for ( long lag = from_step; lag < to_step; ++lag )
  {
    // order is important in this loop, since we have to use the old values
    // (those upon entry to the loop) on right hand sides everywhere
    
    S_.I_tot = P_.k_1 * B_.exc_spikes.get_value(lag) ;
    S_.V_m = V_.P_V * S_.V_m + V_.P_V_input * S_.I_tot;

      

    // send spike, and set spike time in archive.
    set_spiketime( Time::step( slice_origin.get_steps() + lag + 1) );
    //set_spiketime( Time::step( slice_origin.get_steps() + lag) ); //==> did not make any difference
    //set_spiketime( Time::step( slice_origin.get_steps()) ); //==> did not make any difference
     
    
    SpikeEvent se;
    se.set_offset( S_.V_m > 0 ? S_.V_m : 0 );
    kernel().event_delivery_manager.send( *this, se, lag );

    // log membrane potential
    B_.logger_.record_data( slice_origin.get_steps() + lag );
  }
}

void
felixmodule::felix_inh::handle( SpikeEvent& e )
{
  assert( e.get_delay_steps() == 1 );
  assert( e.get_multiplicity() == 1 );
  assert( e.get_rport() == 0 );
  if ( e.get_offset() > 0 )
  {
    // phi == 1 in sender
    B_.exc_spikes.add_value(
        e.get_rel_delivery_steps( kernel().simulation_manager.get_slice_origin() ), e.get_weight() );
  }
}

//void felixmodule::felix_inh::handle(SpikeEvent& e) {
//  assert(e.get_delay_steps() == 0); // Expect 1-step delay
//  assert(e.get_multiplicity() == 1);
//  assert(e.get_rport() == 0);
//  if (e.get_offset() > 0) {
//    // phi == 1 in sender
//    B_.exc_spikes.add_value(
//        e.get_rel_delivery_steps(kernel().simulation_manager.get_slice_origin()),
//        e.get_weight());
//  }
//}



// Do not move this function as inline to h-file. It depends on
// universal_data_logger_impl.h being included here.
void
felixmodule::felix_inh::handle( DataLoggingRequest& e )
{
  B_.logger_.handle( e ); // the logger does this for us
}
