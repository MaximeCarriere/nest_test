/*
 *  felix_spike_recorder.h
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

#ifndef FELIX_SPIKE_RECORDER_H
#define FELIX_SPIKE_RECORDER_H

// C++ includes:
#include <vector>

// Includes from nestkernel:
#include "device_node.h"
#include "event.h"
#include "exceptions.h"
#include "nest_types.h"
#include "recording_device.h"

/* BeginUserDocs: device, recorder, spike

Short description
+++++++++++++++++

Collecting spikes from neurons

Description
+++++++++++

The most universal collector device is the ``felix_spike_recorder``, which
collects and records all *spikes* it receives from neurons that are
connected to it. Each spike received by the spike recorder is
immediately handed over to the selected recording backend for further
processing.

Any node from which spikes are to be recorded, must be connected to
the spike recorder using the standard ``Connect`` command. The
connection ``weights`` and ``delays`` are ignored by the spike
recorder, which means that the spike recorder records the time of
spike creation rather than that of their arrival.

::

   >>> neurons = nest.Create('iaf_psc_alpha', 5)
   >>> sr = nest.Create('felix_spike_recorder')
   >>> nest.Connect(neurons, sr)

The call to ``Connect`` will fail if the connection direction is
reversed (i.e., connecting *sr* to *neurons*).

.. include:: ../models/recording_device.rst

See also
++++++++

Examples using this model
+++++++++++++++++++++++++

.. listexamples:: felix_spike_recorder

EndUserDocs */

namespace felixmodule
{

/**
 * Class felix_spike_recorder
 */

  class felix_spike_recorder : public nest::RecordingDevice
{

public:
  felix_spike_recorder();
  felix_spike_recorder( const felix_spike_recorder& );

  bool
  has_proxies() const override
  {
    return false;
  }

  bool
  local_receiver() const override
  {
    return true;
  }

  Name
  get_element_type() const override
  {
    return nest::names::recorder;
  }

  /**
   * Import sets of overloaded virtual functions.
   * @see Technical Issues / Virtual Functions: Overriding, Overloading, and
   * Hiding
   */
  using nest::Node::handle;
  using nest::Node::handles_test_event;
  using nest::Node::receives_signal;

  void handle( nest::SpikeEvent& ) override;

  size_t handles_test_event( nest::SpikeEvent&, size_t ) override;

  Type get_type() const override;
  nest::SignalType receives_signal() const override;

  void get_status( DictionaryDatum& ) const override;
  void set_status( const DictionaryDatum& ) override;

private:
  void pre_run_hook() override;
  void update( nest::Time const&, const long, const long ) override;
};

inline size_t
  felix_spike_recorder::handles_test_event( nest::SpikeEvent&, size_t receptor_type )
{
  if ( receptor_type != 0 )
  {
    throw nest::UnknownReceptorType( receptor_type, get_name() );
  }
  return 0;
}

 inline nest::SignalType
felix_spike_recorder::receives_signal() const
{
  return nest::ALL;
}

} // namespace

#endif /* #ifndef FELIX_SPIKE_RECORDER_H */
