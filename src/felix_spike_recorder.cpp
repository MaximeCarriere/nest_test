/*
 *  felix_spike_recorder.cpp
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

#include "felix_spike_recorder.h"


// Includes from libnestutil:
#include "compose.hpp"

// Includes from nestkernel:
#include "event_delivery_manager_impl.h"
#include "kernel_manager.h"

// Includes from sli:
#include "dict.h"
#include "dictutils.h"

felixmodule::felix_spike_recorder::felix_spike_recorder()
  : nest::RecordingDevice()
{
}

felixmodule::felix_spike_recorder::felix_spike_recorder( const felix_spike_recorder& n )
  : nest::RecordingDevice( n )
{
}

void
felixmodule::felix_spike_recorder::pre_run_hook()
{
  nest::RecordingDevice::pre_run_hook( nest::RecordingBackend::NO_DOUBLE_VALUE_NAMES, nest::RecordingBackend::NO_LONG_VALUE_NAMES );
}

void
felixmodule::felix_spike_recorder::update( nest::Time const&, const long, const long )
{
  // Nothing to do. Writing to the backend happens in handle().
}

nest::RecordingDevice::Type
felixmodule::felix_spike_recorder::get_type() const
{
  return nest::RecordingDevice::SPIKE_RECORDER;
}

void
felixmodule::felix_spike_recorder::get_status( DictionaryDatum& d ) const
{
  nest::RecordingDevice::get_status( d );

  if ( is_model_prototype() )
  {
    return; // no data to collect
  }

  // if we are the device on thread 0, also get the data from the siblings on other threads
  if ( get_thread() == 0 )
  {
    const std::vector< nest::Node* > siblings = nest::kernel().node_manager.get_thread_siblings( get_node_id() );
    std::vector< nest::Node* >::const_iterator s;
    for ( s = siblings.begin() + 1; s != siblings.end(); ++s )
    {
      ( *s )->get_status( d );
    }
  }
}

void
felixmodule::felix_spike_recorder::set_status( const DictionaryDatum& d )
{
  RecordingDevice::set_status( d );
}

void
felixmodule::felix_spike_recorder::handle( nest::SpikeEvent& e )
{
  // accept spikes only if detector was active when spike was emitted
  if ( is_active( e.get_stamp() ) )
  {
    assert( e.get_multiplicity() == 1 );

    // Record a spike when phi == 1
    if ( e.get_offset() > 0 )
    {
      write( e, nest::RecordingBackend::NO_DOUBLE_VALUES, nest::RecordingBackend::NO_LONG_VALUES );
    }
  }
}
