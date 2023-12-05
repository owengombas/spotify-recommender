interface Props {
    trackID: string;
    height?: number;
}

export default function SpotifyEmbed(props: Props) {
  return (
    <iframe
        src={`https://open.spotify.com/embed/track/${props.trackID}?utm_source=generator`}
        width="100%"
        height={props.height || 352}
        allow="autoplay; clipboard-write; encrypted-media; fullscreen; picture-in-picture"
        loading="lazy"
    />
  )
}
