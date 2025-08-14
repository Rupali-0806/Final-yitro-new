interface ContactCardProps {
  id: number;
  name: string;
  onClick: (id: number) => void;
}

export default function ContactCard({ id, name, onClick }: ContactCardProps) {
  return (
    <div 
      onClick={() => onClick(id)}
      className="aspect-square border-2 border-blue-400 bg-white rounded-lg flex items-center justify-center cursor-pointer hover:bg-blue-50 transition-colors"
    >
      <span className="text-gray-800 font-medium text-lg">{name}</span>
    </div>
  );
}
