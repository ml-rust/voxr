//! Opus as a symphonia decoder: symphonia's demuxers read Opus out of
//! WebM, MKV and Ogg, and `opus-rs`, a pure-Rust decoder, decodes each
//! packet. Opus always decodes at 48 kHz.
//!
//! A stereo stream can carry mono packets, and the other way round, so
//! each packet decodes with a decoder of its own channel count and is
//! mapped onto the stream's: a mono packet plays on both channels, a
//! stereo one is averaged for a mono stream.

use symphonia::core::audio::{
    AsAudioBufferRef, AudioBuffer, AudioBufferRef, Channels, Signal, SignalSpec,
};
use symphonia::core::codecs::{
    CODEC_TYPE_OPUS, CodecDescriptor, CodecParameters, Decoder, DecoderOptions, FinalizeResult,
};
use symphonia::core::errors::{Error, Result, decode_error, unsupported_error};
use symphonia::core::formats::Packet;

/// The rate every Opus stream decodes at.
pub(super) const OPUS_RATE: u32 = 48_000;

/// The longest Opus packet: 120 ms at 48 kHz.
const MAX_FRAMES: usize = 5_760;

/// The channel count `OpusHead` names: its tenth byte.
fn head_channels(extra: Option<&[u8]>) -> Option<usize> {
    extra
        .filter(|h| h.len() >= 19 && h.starts_with(b"OpusHead"))
        .map(|h| usize::from(h[9]))
}

/// The samples at 48 kHz an Opus stream's decoder primes with before its
/// audio starts: `OpusHead`'s pre-skip, else the demuxer's delay.
pub(super) fn pre_skip(params: &CodecParameters) -> usize {
    params
        .extra_data
        .as_deref()
        .filter(|h| h.len() >= 19 && h.starts_with(b"OpusHead"))
        .map(|h| usize::from(u16::from_le_bytes([h[10], h[11]])))
        .or(params.delay.map(|d| d as usize))
        .unwrap_or(0)
}

/// symphonia's decoder for [`CODEC_TYPE_OPUS`], over `opus-rs`.
pub struct OpusDecoder {
    params: CodecParameters,
    channels: usize,
    /// A decoder for mono packets and one for stereo packets, made the
    /// first time a packet of that kind arrives.
    decoders: [Option<opus_rs::OpusDecoder>; 2],
    pcm: Vec<f32>,
    buf: AudioBuffer<f32>,
}

impl OpusDecoder {
    /// The decoder for packets of `channels` channels.
    fn decoder(&mut self, channels: usize) -> Result<&mut opus_rs::OpusDecoder> {
        let slot = &mut self.decoders[channels - 1];
        if slot.is_none() {
            let made = opus_rs::OpusDecoder::new(OPUS_RATE as i32, channels)
                .map_err(Error::DecodeError)?;
            *slot = Some(made);
        }
        slot.as_mut()
            .ok_or(Error::DecodeError("opus: the decoder did not start"))
    }
}

impl Decoder for OpusDecoder {
    fn try_new(params: &CodecParameters, _options: &DecoderOptions) -> Result<Self> {
        let channels = params
            .channels
            .map(|c| c.count())
            .or_else(|| head_channels(params.extra_data.as_deref()))
            .unwrap_or(0);
        let layout = match channels {
            1 => Channels::FRONT_LEFT,
            2 => Channels::FRONT_LEFT | Channels::FRONT_RIGHT,
            _ => return unsupported_error("opus: only mono and stereo streams decode"),
        };
        let mut params = params.clone();
        params.with_sample_rate(OPUS_RATE);
        if params.channels.is_none() {
            params.with_channels(layout);
        }
        Ok(Self {
            params,
            channels,
            decoders: [None, None],
            pcm: vec![0.0; MAX_FRAMES * 2],
            buf: AudioBuffer::new(MAX_FRAMES as u64, SignalSpec::new(OPUS_RATE, layout)),
        })
    }

    fn supported_codecs() -> &'static [CodecDescriptor] {
        &[CodecDescriptor {
            codec: CODEC_TYPE_OPUS,
            short_name: "opus",
            long_name: "Opus",
            inst_func: |params, options| Ok(Box::new(OpusDecoder::try_new(params, options)?)),
        }]
    }

    fn reset(&mut self) {
        self.decoders = [None, None];
    }

    fn codec_params(&self) -> &CodecParameters {
        &self.params
    }

    fn decode(&mut self, packet: &Packet) -> Result<AudioBufferRef<'_>> {
        self.buf.clear();
        let data = packet.buf();
        let Some(&toc) = data.first() else {
            return decode_error("opus: empty packet");
        };
        let packet_channels = if toc & 0x04 != 0 { 2 } else { 1 };
        let mut pcm = std::mem::take(&mut self.pcm);
        let decoded = self.decoder(packet_channels).and_then(|d| {
            d.decode(data, MAX_FRAMES, &mut pcm[..MAX_FRAMES * packet_channels])
                .map_err(Error::DecodeError)
        });
        let frames = match decoded {
            Ok(n) => n.min(MAX_FRAMES),
            Err(e) => {
                self.pcm = pcm;
                return Err(e);
            }
        };
        self.buf.render_reserved(Some(frames));
        for c in 0..self.channels {
            let out = self.buf.chan_mut(c);
            for (i, v) in out.iter_mut().enumerate().take(frames) {
                *v = match (packet_channels, self.channels) {
                    (1, _) => pcm[i],
                    (2, 1) => 0.5 * (pcm[2 * i] + pcm[2 * i + 1]),
                    _ => pcm[2 * i + c],
                };
            }
        }
        self.pcm = pcm;
        Ok(self.buf.as_audio_buffer_ref())
    }

    fn finalize(&mut self) -> FinalizeResult {
        FinalizeResult::default()
    }

    fn last_decoded(&self) -> AudioBufferRef<'_> {
        self.buf.as_audio_buffer_ref()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn head(channels: u8, pre_skip: u16) -> Box<[u8]> {
        let mut h = b"OpusHead".to_vec();
        h.push(1);
        h.push(channels);
        h.extend_from_slice(&pre_skip.to_le_bytes());
        h.extend_from_slice(&48_000u32.to_le_bytes());
        h.extend_from_slice(&[0, 0, 0]);
        h.into_boxed_slice()
    }

    #[test]
    fn the_head_names_channels_and_pre_skip() {
        let mut params = CodecParameters::new();
        params
            .for_codec(CODEC_TYPE_OPUS)
            .with_extra_data(head(2, 312));
        assert_eq!(pre_skip(&params), 312);
        let made = OpusDecoder::try_new(&params, &DecoderOptions::default());
        assert!(made.is_ok_and(|d| d.channels == 2));
        let mut bare = CodecParameters::new();
        bare.for_codec(CODEC_TYPE_OPUS).with_delay(80);
        assert_eq!(pre_skip(&bare), 80);
        assert!(OpusDecoder::try_new(&bare, &DecoderOptions::default()).is_err());
    }
}
